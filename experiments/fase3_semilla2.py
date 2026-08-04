# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

# =============================================================================
# Omega-S : Fase 2 Segunda Semilla (SEED=123)
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================
"""
FASE 3C: Segunda semilla para validación estadística
=====================================================
Replica el experimento de olvido catastrófico de Fase 2 con SEED=123.
Corre baseline y Omega-S secuencialmente en una sola GPU.

Ejecutar con: CUDA_VISIBLE_DEVICES=1 python fase3_semilla2.py
"""

import json, torch, numpy as np, signal
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED          = 123   # semilla diferente a la Fase 2 (que usó 42)
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
MAX_SAMPLES   = 5000
HUMANEVAL_N   = 164
MAX_NEW_TOK   = 256
OMEGA_LAMBDA  = 0.05
OMEGA_EVERY_K = 10
OMEGA_PROBES  = 3
OUT_FILE      = "results_seed2.json"

torch.manual_seed(SEED)
np.random.seed(SEED)

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
    model.print_trainable_parameters()
    return model, tok

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

def omega_pen(model):
    total, count = 0.0, 0
    for name, p in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and p.requires_grad:
            total += hutchinson_tr_a3(p.data)
            count += p.numel()
    return OMEGA_LAMBDA * total / max(count, 1)

def train_epoch(model, loader, optimizer, use_omega, step_offset=0):
    model.train()
    total, step = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        mask = batch.get("attention_mask",
                          torch.ones_like(ids)).to(DEVICE)
        loss = model(input_ids=ids, attention_mask=mask,
                     labels=labs).loss / GRAD_ACCUM
        if use_omega and (step + step_offset) % OMEGA_EVERY_K == 0:
            loss = loss + omega_pen(model) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        total += loss.item() * GRAD_ACCUM; step += 1
        if step % 100 == 0:
            print(f"  step {step}/{len(loader)} loss={total/step:.4f}")
    return total / max(step, 1)

def train_domain(model, loader, epochs, use_omega, tag):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    print(f"\nEntrenando {tag}...")
    for e in range(epochs):
        loss = train_epoch(model, loader, opt, use_omega)
        print(f"  Epoch {e+1}/{epochs} loss={loss:.4f}")
    return model

def humaneval(model, tok, n=HUMANEVAL_N, tag=""):
    print(f"\nHumanEval {tag} - {n} problemas...")
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

def run_condition(use_omega, label):
    print(f"\n{'#'*60}\n{label} (SEED={SEED})\n{'#'*60}")
    torch.manual_seed(SEED)
    model, tok = load_model()
    code_loader  = get_loader(tok, "code", MAX_SAMPLES)
    prose_loader = get_loader(tok, "prose", MAX_SAMPLES)
    res = {"label": label, "seed": SEED, "use_omega": use_omega}
    res["he_baseline"] = humaneval(model, tok, HUMANEVAL_N, "baseline")
    model = train_domain(model, code_loader, EPOCHS_A, use_omega,
                         "Dominio-A codigo")
    res["he_post_code"] = humaneval(model, tok, HUMANEVAL_N, "post-codigo")
    model = train_domain(model, prose_loader, EPOCHS_B, use_omega,
                         "Dominio-B prosa")
    res["he_post_prose"] = humaneval(model, tok, HUMANEVAL_N, "post-prosa")
    res["forgetting"]    = res["he_post_code"] - res["he_post_prose"]
    res["retention_pct"] = res["he_post_prose"] / max(res["he_post_code"], 1e-6)
    print(f"\n[{label}] Retencion: {res['retention_pct']:.4%} | "
          f"Olvido: {res['forgetting']:.4%}")
    return res

if __name__ == "__main__":
    print("=" * 60)
    print(f"FASE 3C: SEGUNDA SEMILLA (SEED={SEED})")
    print("=" * 60)

    res_base  = run_condition(False, "Baseline_LoRA_S2")
    res_omega = run_condition(True,  "Omega_S_LoRA_S2")

    # Comparar con Semilla 1 (SEED=42)
    s1_base_ret  = 0.8387
    s1_omega_ret = 0.8667
    s2_base_ret  = res_base["retention_pct"]
    s2_omega_ret = res_omega["retention_pct"]

    print("\n" + "=" * 65)
    print("VALIDACION ESTADISTICA (2 semillas)")
    print("=" * 65)
    print(f"{'Condicion':<25} {'Semilla 1':>10} {'Semilla 2':>10} {'Media':>8}")
    print("-" * 55)
    print(f"{'Baseline retencion':<25} {s1_base_ret:>10.2%} "
          f"{s2_base_ret:>10.2%} {(s1_base_ret+s2_base_ret)/2:>8.2%}")
    print(f"{'Omega-S retencion':<25} {s1_omega_ret:>10.2%} "
          f"{s2_omega_ret:>10.2%} {(s1_omega_ret+s2_omega_ret)/2:>8.2%}")
    delta_s1 = s1_omega_ret - s1_base_ret
    delta_s2 = s2_omega_ret - s2_base_ret
    print(f"{'Delta (Omega-Base)':<25} {delta_s1:>+10.2%} "
          f"{delta_s2:>+10.2%} {(delta_s1+delta_s2)/2:>+8.2%}")
    print("=" * 65)

    if delta_s1 > 0 and delta_s2 > 0:
        print("CONCLUSION: La mejora de retencion de Omega-S es")
        print("CONSISTENTE entre semillas. Resultado robusto.")
    else:
        print("CONCLUSION: Resultado inconsistente entre semillas.")
        print("Considerar mas semillas antes de publicar.")

    results = {
        "seed": SEED,
        "baseline": res_base,
        "omega_s": res_omega,
        "seed1_comparison": {
            "baseline_s1": s1_base_ret,
            "omega_s1": s1_omega_ret,
            "delta_s1": delta_s1,
            "delta_s2": delta_s2,
            "mean_delta": (delta_s1 + delta_s2) / 2,
        }
    }
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {OUT_FILE}")
