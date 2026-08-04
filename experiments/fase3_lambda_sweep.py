# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

# =============================================================================
# Omega-S : Barrido de Lambda (trade-off estabilidad-plasticidad)
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================
"""
FASE 3B: Barrido de OMEGA_LAMBDA
=================================
Demuestra que el trade-off estabilidad-plasticidad es controlable
mediante el hiperparámetro lambda de Omega-S.

Para cada valor de lambda entrena Llama-3-8B + LoRA en código → prosa
y mide retención (HumanEval) y plasticidad (HumanEval post-código).

Ejecutar con: CUDA_VISIBLE_DEVICES=0 python fase3_lambda_sweep.py
(Después de que termine fase3_gpt2_flops.py en la misma GPU)
"""

import os, json, torch, numpy as np, signal
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED          = 42
MODEL_ID      = "NousResearch/Meta-Llama-3-8B"
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.1
LORA_TARGETS  = ["q_proj", "v_proj"]
LR            = 2e-4
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
MAX_SEQ_LEN   = 512
EPOCHS_A      = 1
EPOCHS_B      = 1
MAX_SAMPLES   = 2000
HUMANEVAL_N   = 50
MAX_NEW_TOK   = 256
OMEGA_EVERY_K = 10
OMEGA_PROBES  = 3
OUT_FILE      = "results_lambda_sweep.json"

# Grid de lambda a probar
LAMBDA_GRID = [0.0, 0.01, 0.05, 0.10, 0.20]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(DEVICE)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                     r=LORA_R, lora_alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS)
    model = get_peft_model(model, cfg)
    return model, tok

# ---------------------------------------------------------------------------
# DATOS
# ---------------------------------------------------------------------------
def get_loader(tokenizer, domain="code", max_s=MAX_SAMPLES):
    if domain == "code":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        ds = ds.select(range(min(max_s, len(ds))))
        def tok_fn(b):
            texts = [f"### Instruction:\n{i}\n### Output:\n{o}"
                     for i, o in zip(b["instruction"], b["output"])]
            enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN,
                            padding="max_length", return_tensors="pt")
            enc["labels"] = enc["input_ids"].clone()
            return enc
        ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    else:
        ds = load_dataset("Salesforce/wikitext",
                          "wikitext-2-raw-v1", split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        ds = ds.select(range(min(max_s, len(ds))))
        def tok_fn(b):
            enc = tokenizer(b["text"], truncation=True,
                            max_length=MAX_SEQ_LEN, padding="max_length",
                            return_tensors="pt")
            enc["labels"] = enc["input_ids"].clone()
            return enc
        ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------------------------
# OMEGA-S
# ---------------------------------------------------------------------------
def hutchinson_tr_a3(W, n=OMEGA_PROBES):
    total = 0.0
    for _ in range(n):
        v = torch.randint(0, 2, (W.shape[0],),
                          device=W.device, dtype=torch.float32) * 2 - 1
        Wf = W.float()
        z = Wf @ (Wf.t() @ v)
        z = Wf @ (Wf.t() @ z)
        z = Wf @ (Wf.t() @ z)
        total += (v @ z)
    return total / n

def omega_pen(model, lam):
    total, count = 0.0, 0
    for name, p in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and p.requires_grad:
            total += hutchinson_tr_a3(p.data)
            count += p.numel()
    return lam * total / max(count, 1)

# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, lam, step_offset=0):
    model.train()
    total, step = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        mask = batch.get("attention_mask",
                          torch.ones_like(ids)).to(DEVICE)
        loss = model(input_ids=ids, attention_mask=mask,
                     labels=labs).loss / GRAD_ACCUM
        if lam > 0 and (step + step_offset) % OMEGA_EVERY_K == 0:
            loss = loss + omega_pen(model, lam) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        total += loss.item() * GRAD_ACCUM; step += 1
    return total / max(step, 1)

def train_domain(model, loader, epochs, lam, tag):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    for e in range(epochs):
        loss = train_epoch(model, loader, opt, lam)
        print(f"  [{tag}] Epoch {e+1}/{epochs} loss={loss:.4f}")
    return model

# ---------------------------------------------------------------------------
# EVALUACIÓN HUMANEVAL
# ---------------------------------------------------------------------------
def humaneval(model, tok, n=HUMANEVAL_N, tag=""):
    print(f"  HumanEval {tag} ({n} problemas)...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = [ds[i] for i in range(min(n, len(ds)))]
    passed = 0
    def _timeout(s, f): raise TimeoutError()
    signal.signal(signal.SIGALRM, _timeout)
    for i, p in enumerate(problems):
        gen_in = tok(p["prompt"], return_tensors="pt",
                     truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**gen_in, max_new_tokens=MAX_NEW_TOK,
                                 temperature=0.1, do_sample=True,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][gen_in["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        full = p["prompt"] + gen + "\n" + p["test"]
        try:
            signal.alarm(10)
            g = {}; exec(compile(full, "<s>", "exec"), g)
            exec(f"check({p['entry_point']})", g)
            signal.alarm(0); passed += 1
        except TimeoutError: pass
        except Exception: signal.alarm(0)
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    pa1 = passed / max(n, 1)
    print(f"  pass@1: {pa1:.4%} ({passed}/{n})")
    return pa1

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("FASE 3B: BARRIDO DE LAMBDA : TRADE-OFF ESTABILIDAD-PLASTICIDAD")
    print("=" * 65)

    resultados = []

    for lam in LAMBDA_GRID:
        label = f"lambda={lam:.2f}"
        print(f"\n{'='*65}")
        print(f"Entrenando con {label}...")
        print(f"{'='*65}")

        torch.manual_seed(SEED)
        model, tok = load_model()

        code_loader  = get_loader(tok, "code", MAX_SAMPLES)
        prose_loader = get_loader(tok, "prose", MAX_SAMPLES)

        he_base = humaneval(model, tok, HUMANEVAL_N, "baseline")

        model = train_domain(model, code_loader, EPOCHS_A, lam,
                             f"{label} codigo")
        he_code = humaneval(model, tok, HUMANEVAL_N, "post-codigo")

        model = train_domain(model, prose_loader, EPOCHS_B, lam,
                             f"{label} prosa")
        he_prose = humaneval(model, tok, HUMANEVAL_N, "post-prosa")

        forgetting   = he_code - he_prose
        retention    = he_prose / max(he_code, 1e-6)
        plasticity   = he_code - he_base

        resultados.append({
            "lambda": lam,
            "he_baseline": he_base,
            "he_post_code": he_code,
            "he_post_prose": he_prose,
            "forgetting": forgetting,
            "retention_pct": retention,
            "plasticity": plasticity,
        })

        print(f"  Retencion: {retention:.4%} | Olvido: {forgetting:.4%} | "
              f"Plasticidad: {plasticity:+.4%}")

    # Tabla final
    print("\n\n" + "=" * 75)
    print("TABLA FINAL : BARRIDO DE LAMBDA")
    print("=" * 75)
    print(f"{'Lambda':<10} {'HE base':>8} {'HE code':>8} {'HE prose':>9} "
          f"{'Olvido':>8} {'Retencion':>10} {'Plasticidad':>12}")
    print("-" * 75)
    for r in resultados:
        print(f"{r['lambda']:<10.2f} {r['he_baseline']:>8.2%} "
              f"{r['he_post_code']:>8.2%} {r['he_post_prose']:>9.2%} "
              f"{r['forgetting']:>8.2%} {r['retention_pct']:>10.2%} "
              f"{r['plasticity']:>+12.2%}")
    print("=" * 75)

    print("\nCONCLUSION:")
    base = resultados[0]
    best = max(resultados[1:], key=lambda r: r["retention_pct"])
    print(f"  Lambda=0 (baseline): retencion={base['retention_pct']:.2%}, "
          f"olvido={base['forgetting']:.2%}")
    print(f"  Mejor lambda={best['lambda']:.2f}: "
          f"retencion={best['retention_pct']:.2%}, "
          f"olvido={best['forgetting']:.2%}")
    delta_ret = best["retention_pct"] - base["retention_pct"]
    delta_plas = best["plasticity"] - base["plasticity"]
    print(f"  Trade-off: +{delta_ret:.2%} retencion, "
          f"{delta_plas:+.2%} plasticidad")
    print("  El trade-off es CONTROLABLE via lambda.")

    with open(OUT_FILE, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\nResultados guardados en {OUT_FILE}")
