# =============================================================================
# Omega-S : Re-corrida de retención (RunPod)
# Sustituye a fase2_omega.py y fase3_semilla*.py
# =============================================================================
# Por qué existe: en los scripts originales la penalización se calculaba con
# `hutchinson_tr_a3(p.data)`. `.data` desacopla del grafo de autograd, así que
# el término se sumaba a la pérdida como una CONSTANTE y su gradiente era CERO.
# El regularizador nunca tocó los pesos. Ver AUDIT.md.
#
# Qué cambia:
#   1. Penalización CONECTADA, sobre el peso efectivo W_base + s*(B@A).
#   2. Verificación de gradiente ANTES de entrenar. Si es cero, aborta.
#   3. RNG DEDICADO para la penalización. Antes consumía el RNG global y
#      desincronizaba el brazo Omega respecto al baseline: era otra semilla,
#      no otro método. Ese era el confound que producía las diferencias.
#   4. Semillas de verdad distintas: 42, 123, 456. Antes fase3_semilla3.py era
#      copia de fase3_semilla2.py y ambos tenían SEED=123.
#   5. CINCO brazos, incluidos los baselines que faltaban:
#        none      sin regularizar
#        wd        weight decay AJUSTADO por rejilla, no un valor por defecto
#        ewc       Elastic Weight Consolidation, baseline canónico desde 2017
#        omega_raw Tr((WW^T)^3) crudo, el de los experimentos originales
#        omega_lib StochasticOmegaS del paquete, el que describen el preprint
#                  y la patente. Correr ambos contesta de paso si librería y
#                  experimentos deben unificarse.
#   6. HumanEval pass@1, la misma métrica del número publicado (83.03/81.07).
#   7. OMEGA_LAMBDA se CALIBRA. El 0.05 anterior se ajustó contra una constante.
#
# Uso:
#   python rerun_retention.py --smoke                            # ~10 min
#   python rerun_retention.py --cell --seed 42 --arm omega_raw   # UNA celda
#   python rerun_retention.py --all                              # rejilla
# =============================================================================

import os, sys, json, time, argparse, statistics, itertools, subprocess, tempfile
import torch, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset

DECISION_RULE = """
REGLA DE DECISION, PRERREGISTRADA (escrita antes de mirar ningun resultado)

Omega-S se considera VALIDADO si y solo si:
  (a) supera a weight decay AJUSTADO en retencion en >= 2 de 3 semillas,
  (b) sin pagar mas de 1pp adicional de plasticidad frente a ese mismo brazo,
  (c) y queda dentro de una desviacion tipica de EWC o por encima.
Si no se cumple, el claim de retencion NO se publica y el preprint de LLM se
reescribe sin el. Vale por separado para omega_raw y para omega_lib.
"""

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS    = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
ARMS     = ["none", "wd", "ewc", "rownorm", "omega_raw", "omega_lib"]
WD_GRID  = [0.0, 0.01, 0.05, 0.1]

MODEL_ID     = "NousResearch/Meta-Llama-3-8B"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1
LORA_TARGETS = ["q_proj", "v_proj"]
LR, BATCH_SIZE, GRAD_ACCUM = 2e-4, 2, 4
MAX_SEQ_LEN, MAX_SAMPLES   = 512, 5000
EPOCHS_A = EPOCHS_B = 1

HUMANEVAL_N  = 164
MAX_NEW_TOK  = 256
GEN_TEMP     = 0.1
HE_BATCH     = 32      # generacion por lotes: 8-10x mas rapido que de uno en uno

OMEGA_EVERY_K = 10
OMEGA_PROBES  = 16     # 3 daba un estimador dominado por ruido de muestreo
OMEGA_MODULES = 8      # muestreo de modulos; W W^T cuesta O(N^2 * in)
EWC_LAMBDA    = float(os.environ.get("EWC_LAMBDA", "1000.0"))  # barrible


# ===========================================================================
# PENALIZACIONES
# ===========================================================================
def iter_effective_weights(model):
    """W_eff = W_base + scaling * (B @ A). Conectado al grafo."""
    for name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        base = getattr(mod, "base_layer", None)
        if base is None or not hasattr(base, "weight"):
            continue
        for key in mod.lora_A.keys():
            A = mod.lora_A[key].weight
            B = mod.lora_B[key].weight
            s = mod.scaling[key] if isinstance(mod.scaling, dict) else mod.scaling
            delta = s * (B @ A)
            W = base.weight
            if W.shape != delta.shape and tuple(W.shape) == tuple(delta.shape)[::-1]:
                delta = delta.t()
            yield name, W + delta
            break


def hutchinson_tr_a3(W, n, gen):
    """Tr((W W^T)^3). El metodo de los experimentos originales."""
    total, Wf = 0.0, W.float()
    for _ in range(n):
        v = (torch.randint(0, 2, (Wf.shape[0],), generator=gen,
                           dtype=torch.float32) * 2 - 1).to(Wf.device)
        z = Wf @ (Wf.t() @ v)
        z = Wf @ (Wf.t() @ z)
        z = Wf @ (Wf.t() @ z)
        total = total + (v @ z)
    return total / n


_LIB_CORE = None
def omega_lib_core():
    """StochasticOmegaS del paquete: log((M*Coex)/(C*D)) sobre sigmoid(|W W^T|)."""
    global _LIB_CORE
    if _LIB_CORE is None:
        try:
            from omega_s import StochasticOmegaS
        except ImportError:
            from omega_s.omega_s import StochasticOmegaS
        _LIB_CORE = StochasticOmegaS(num_samples=OMEGA_PROBES).to(DEVICE)
    return _LIB_CORE


def omega_pen(model, arm, gen):
    mods = list(iter_effective_weights(model))
    if not mods:
        raise RuntimeError("Sin modulos LoRA. Revisa LORA_TARGETS.")
    k = min(OMEGA_MODULES, len(mods))
    idx = torch.randperm(len(mods), generator=gen)[:k].tolist()
    if arm == "omega_raw":
        total = sum(hutchinson_tr_a3(mods[i][1], OMEGA_PROBES, gen) for i in idx)
    else:
        core = omega_lib_core()
        total = sum(core(mods[i][1]) for i in idx)
    return total / k


def rownorm_pen(model, gen):
    """Control: penaliza SOLO la varianza de las normas de fila del peso
    efectivo. Cero topologia. Si Omega solo iguala normas, este brazo lo
    replica. Mismo muestreo de modulos y mismo RNG que omega_pen para que
    la comparacion sea limpia."""
    mods = list(iter_effective_weights(model))
    if not mods:
        raise RuntimeError("Sin modulos LoRA.")
    k = min(OMEGA_MODULES, len(mods))
    idx = torch.randperm(len(mods), generator=gen)[:k].tolist()
    total = sum(mods[i][1].float().norm(dim=1).var() for i in idx)
    return total / k


def _pen_dispatch(model, arm, gen):
    return rownorm_pen(model, gen) if arm == "rownorm" else omega_pen(model, arm, gen)


def assert_connected(model, arm, gen):
    """La red de seguridad que no existia. Falla ruidosamente si no hay gradiente."""
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params: p.grad = None
    pen = _pen_dispatch(model, arm, gen)
    if not hasattr(pen, "grad_fn") or pen.grad_fn is None:
        raise RuntimeError("PENALIZACION DESCONECTADA (grad_fn=None). Busca un .data o .item().")
    pen.backward()
    g = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
    for p in params: p.grad = None
    if not torch.isfinite(g) or g.item() == 0.0:
        raise RuntimeError(f"GRADIENTE NULO ({g}). NO entrenes.")
    print(f"  [{arm}] conectada. valor={pen.item():+.4e}  ||grad||={g.item():.4e}")
    return g.item()


def calibrate_lambda(model, batch, arm, gen, target=None, n=5):
    if target is None:
        target = float(os.environ.get("OMEGA_TARGET", "0.1"))
    """El OMEGA_LAMBDA=0.05 anterior se ajusto contra una constante. No vale."""
    params = [p for p in model.parameters() if p.requires_grad]
    ratios = []
    for _ in range(n):
        for p in params: p.grad = None
        model(**batch).loss.backward()
        g_ce = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
        for p in params: p.grad = None
        _pen_dispatch(model, arm, gen).backward()
        g_om = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
        ratios.append((g_om / (g_ce + 1e-12)).item())
    for p in params: p.grad = None
    lam = target / (statistics.mean(ratios) + 1e-12)
    print(f"  [{arm}] ratio grad={statistics.mean(ratios):.3e} -> lambda={lam:.4e}")
    return lam


def compute_fisher(model, loader, n_batches=32):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval(); i = -1
    for i, b in enumerate(loader):
        if i >= n_batches: break
        model.zero_grad()
        model(input_ids=b["input_ids"].to(DEVICE),
              labels=b["labels"].to(DEVICE)).loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher: fisher[n] /= max(min(i + 1, n_batches), 1)
    star = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    return fisher, star


def ewc_pen(model, fisher, star):
    return sum((fisher[n] * (p - star[n]) ** 2).sum()
               for n, p in model.named_parameters() if n in fisher)


# ===========================================================================
# HUMANEVAL
# ===========================================================================
_SANDBOX = 'import sys, io, contextlib\n' \
           'src = open(sys.argv[1]).read()\n' \
           'buf = io.StringIO()\n' \
           'try:\n' \
           '    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n' \
           '        exec(src, {"__name__": "__main__"})\n' \
           '    print("PASS")\n' \
           'except BaseException:\n' \
           '    print("FAIL")\n'


def _run_sandboxed(code, timeout=10):
    """Ejecuta en SUBPROCESO. El exec() in-process del script original podia
    tumbar la corrida entera con el codigo generado por el modelo."""
    with tempfile.TemporaryDirectory() as d:
        cand = os.path.join(d, "cand.py"); runner = os.path.join(d, "run.py")
        open(cand, "w").write(code); open(runner, "w").write(_SANDBOX)
        try:
            r = subprocess.run([sys.executable, runner, cand], timeout=timeout,
                               capture_output=True, text=True)
            return "PASS" in r.stdout
        except Exception:
            return False


@torch.no_grad()
def humaneval(model, tok, n=HUMANEVAL_N, tag="", batch_size=None):
    """
    HumanEval pass@1 con generacion POR LOTES.

    El script original generaba de uno en uno: 164 problemas x 2 evaluaciones
    x ~10 s por generacion = ~55 min por celda, o sea el 70% del coste total.
    Por lotes de 32 baja a ~6 min. Mismo resultado, 8-10x mas rapido.

    El tokenizer usa padding_side="left", que es lo correcto para generar con
    modelos decoder-only: el padding queda ANTES del prompt y no contamina.
    """
    bs = batch_size or HE_BATCH
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    problems = [ds[i] for i in range(min(n, len(ds)))]
    passed, t0 = 0, time.time()
    model.eval()

    for start in range(0, len(problems), bs):
        chunk = problems[start:start + bs]
        enc = tok([p["prompt"] for p in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOK,
                             temperature=GEN_TEMP, do_sample=True,
                             pad_token_id=tok.eos_token_id)
        gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        for p, g in zip(chunk, gens):
            code = p["prompt"] + g + "\n" + p["test"] + "\ncheck(" + p["entry_point"] + ")\n"
            passed += _run_sandboxed(code)
        done = min(start + bs, len(problems))
        print("    [" + tag + "] " + str(done) + "/" + str(len(problems)) +
              " pass@1=" + format(passed/done, ".3f") +
              " (" + format(time.time()-t0, ".0f") + "s)")
    return passed / max(len(problems), 1)


# ===========================================================================
# DATOS / MODELO
# ===========================================================================
def load_model(seed, smoke):
    torch.manual_seed(seed); np.random.seed(seed)
    mid = "sshleifer/tiny-gpt2" if smoke else MODEL_ID
    tok = AutoTokenizer.from_pretrained(mid)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16 if not smoke else torch.float32,
        low_cpu_mem_usage=True).to(DEVICE)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                     r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=["c_attn"] if smoke else LORA_TARGETS)
    return get_peft_model(m, cfg), tok


def _tok(tok, texts):
    enc = tok(texts, truncation=True, max_length=MAX_SEQ_LEN,
              padding="max_length", return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


def get_loader(tok, domain, smoke):
    n = 32 if smoke else MAX_SAMPLES
    if domain == "code":
        ds = load_dataset("code-search-net/code_search_net", "python", split="train",
                          trust_remote_code=True).select(range(n))
        def f(b):
            texts = ["### Docstring:\n" + d + "\n### Code:\n" + c
                     for d, c in zip(b["func_documentation_string"],
                                     b["whole_func_string"])]
            return _tok(tok, texts)
        ds = ds.map(f, batched=True, remove_columns=ds.column_names)
    else:
        # STREAMING: openwebtext son ~55 GB si se descarga entero y solo
        # necesitamos MAX_SAMPLES muestras. Con streaming no toca disco.
        it = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                          trust_remote_code=True).take(n)
        rows = [r["text"] for r in it]
        ds = Dataset.from_dict({"text": rows})
        ds = ds.map(lambda b: _tok(tok, b["text"]), batched=True,
                    remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


# ===========================================================================
# ENTRENAMIENTO
# ===========================================================================
def train_domain(model, loader, arm, wd, lam=None, gen=None, fisher=None, star=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=wd if arm == "wd" else 0.0)
    model.train()
    for step, b in enumerate(loader):
        loss = model(input_ids=b["input_ids"].to(DEVICE),
                     labels=b["labels"].to(DEVICE)).loss / GRAD_ACCUM
        if arm.startswith("omega") and step % OMEGA_EVERY_K == 0:
            loss = loss + lam * omega_pen(model, arm, gen) / GRAD_ACCUM
        if arm == "rownorm" and step % OMEGA_EVERY_K == 0:
            loss = loss + lam * rownorm_pen(model, gen) / GRAD_ACCUM
        if arm == "ewc" and fisher is not None:
            loss = loss + EWC_LAMBDA * ewc_pen(model, fisher, star) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
    return model


def run_cell(seed, arm, wd, smoke, he_n):
    t0 = time.time()
    print("\n" + "="*66 + "\nsemilla=" + str(seed) + "  brazo=" + arm +
          "  wd=" + str(wd) + "\n" + "="*66)
    model, tok = load_model(seed, smoke)
    code_tr  = get_loader(tok, "code", smoke)
    prose_tr = get_loader(tok, "prose", smoke)

    # RNG DEDICADO. Sin esto, la penalizacion consume el flujo global y el
    # brazo deja de ser comparable con el baseline: era el confound original.
    gen = torch.Generator().manual_seed(seed + 10_000)

    lam = None
    if arm.startswith("omega") or arm == "rownorm":
        assert_connected(model, arm, gen)
        b0 = next(iter(code_tr))
        lam = calibrate_lambda(model, {"input_ids": b0["input_ids"].to(DEVICE),
                                       "labels": b0["labels"].to(DEVICE)}, arm, gen)

    model = train_domain(model, code_tr, arm, wd, lam, gen)
    he_A = humaneval(model, tok, he_n, "tras A")
    print("  HumanEval pass@1 tras tarea A: " + format(he_A, ".4f"))

    fisher = star = None
    if arm == "ewc":
        fisher, star = compute_fisher(model, code_tr)

    model = train_domain(model, prose_tr, arm, wd, lam, gen, fisher, star)
    he_B = humaneval(model, tok, he_n, "tras B")
    print("  HumanEval pass@1 tras tarea B: " + format(he_B, ".4f"))

    ret = he_B / he_A if he_A > 0 else float("nan")
    print("  RETENCION: " + format(ret, ".4f") +
          "   (" + format(time.time()-t0, ".0f") + "s)")

    return dict(seed=seed, arm=arm, wd=wd, omega_lambda=lam,
                omega_target=float(os.environ.get("OMEGA_TARGET", "0.1"))
                             if (arm.startswith("omega") or arm=="rownorm") else None,
                humaneval_after_A=he_A, humaneval_after_B=he_B,
                retention_pct=ret, seconds=time.time() - t0, smoke=smoke)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell",  action="store_true", help="una celda, para cronometrar")
    ap.add_argument("--all",   action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm",  default="omega_raw", choices=ARMS)
    ap.add_argument("--he-n", type=int, default=None)
    ap.add_argument("--out",  default="rerun_results.json")
    ap.add_argument("--cells", default=None,
                    help="indices de celda separados por comas, p.ej. 0,3,7. "
                         "Lo usa launch_parallel.sh para repartir entre GPUs.")
    ap.add_argument("--best-wd", type=float, default=None,
                    help="salta la rejilla de weight decay y usa este valor")
    ap.add_argument("--he-batch", type=int, default=None)
    ap.add_argument("--omega-sweep", default=None,
                    help="targets de calibracion separados por comas, p.ej. 0.03,0.1,0.3,0.5")
    ap.add_argument("--omega-arm", default="omega_lib", choices=["omega_raw","omega_lib","rownorm"],
                    help="que brazo barrer con --omega-sweep")
    ap.add_argument("--ewc-sweep", default=None,
                    help="lambdas de EWC separados por comas, p.ej. 100,500,1000,5000,10000. Corre EWC con cada uno en la semilla 42 para ajustarlo antes de la corrida grande.")
    a = ap.parse_args()
    if a.he_batch:
        global HE_BATCH
        HE_BATCH = a.he_batch

    he_n = a.he_n or (8 if a.smoke else HUMANEVAL_N)
    print(DECISION_RULE)
    rows = []

    if a.omega_sweep is not None:
        # Barrido del objetivo de calibracion de omega (y rownorm). Ajustar
        # omega con la misma vara que wd y ewc: sin esto, la comparacion
        # favorece a quien se ajusta. target = fraccion del gradiente de la
        # tarea que ocupa la penalizacion (3%, 10%, 30%, 50%).
        import os as _os
        which = a.omega_arm
        targets = [float(x) for x in a.omega_sweep.split(",")]
        wd = a.best_wd if a.best_wd is not None else 0.0
        for tg in targets:
            _os.environ["OMEGA_TARGET"] = str(tg)
            print("\n### " + which + " target = " + str(tg) + " ###")
            r = run_cell(SEEDS[0], which, wd, a.smoke, he_n)
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=2)
        best = max(rows, key=lambda r: r["retention_pct"])
        print("\n>>> Mejor " + which + ": target=" + str(best["omega_target"]) +
              " con retencion " + format(best["retention_pct"], ".4f"))
        print("Guardado en " + a.out)
        return

    if a.ewc_sweep is not None:
        # Barrido de EWC en una semilla para ajustar su lambda. EWC salio
        # rindiendo como "no hacer nada" (0.601 vs 0.605), lo que sugiere que
        # su lambda por defecto (1000) esta mal puesto. Esto lo comprueba.
        import os as _os
        lams = [float(x) for x in a.ewc_sweep.split(",")]
        wd = a.best_wd if a.best_wd is not None else 0.0
        for lam in lams:
            _os.environ["EWC_LAMBDA"] = str(lam)
            print("\n### EWC lambda = " + str(lam) + " ###")
            r = run_cell(SEEDS[0], "ewc", wd, a.smoke, he_n)
            r["ewc_lambda"] = lam
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=2)
        best = max(rows, key=lambda r: r["retention_pct"])
        print("\n>>> Mejor EWC: lambda=" + str(best["ewc_lambda"]) +
              " con retencion " + format(best["retention_pct"], ".4f"))
        print("Guardado en " + a.out)
        return

    if a.cells is not None:
        # Reparto para ejecucion paralela. El orden de CELLS es fijo y
        # deterministico, asi que cada GPU sabe cuales le tocan sin coordinarse.
        CELLS = [(sd, ar) for sd in SEEDS for ar in ARMS]
        wd = a.best_wd if a.best_wd is not None else 0.0
        for i in [int(x) for x in a.cells.split(",")]:
            sd, ar = CELLS[i]
            print("\n### celda " + str(i) + " de " + str(len(CELLS)) + " ###")
            rows.append(run_cell(sd, ar, wd, a.smoke, he_n))
            json.dump(rows, open(a.out, "w"), indent=2)
        json.dump(rows, open(a.out, "w"), indent=2)
        print("\nGuardado en " + a.out)
        return

    if a.all:
        print("\n### Paso 0: ajustar weight decay ###")
        wd_rows = [run_cell(SEEDS[0], "wd", w, a.smoke, he_n) for w in WD_GRID]
        best_wd = max(wd_rows, key=lambda r: r["retention_pct"])["wd"]
        print("\n>>> weight decay ajustado: " + str(best_wd))
        rows += wd_rows
        for seed, arm in itertools.product(SEEDS, ARMS):
            rows.append(run_cell(seed, arm, best_wd, a.smoke, he_n))
            json.dump(rows, open(a.out, "w"), indent=2)   # guardar tras cada celda
    else:
        _wd = a.best_wd if a.best_wd is not None else 0.0
        rows.append(run_cell(a.seed, a.arm, _wd, a.smoke, he_n))

    json.dump(rows, open(a.out, "w"), indent=2)
    print("\nGuardado en " + a.out)
    if a.cell:
        s = rows[0]["seconds"]
        print("\nUna celda = " + format(s/60, ".1f") + " min. Rejilla completa "
              "(19 celdas) ~ " + format(19*s/3600, ".1f") + " h de GPU.")


if __name__ == "__main__":
    main()
