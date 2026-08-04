# =============================================================================
# Omega-S : Reproducibility Script (Fase 3)
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

# =============================================================================
# Omega-S : Experimento FLOPs en Transformer Real (GPT-2)
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================
"""
FASE 3A: FLOPs en transformer real (GPT-2 completo, sin LoRA)
==============================================================
Objetivo: demostrar que Omega-S + group-lasso produce sparsity estructurada
en las capas de atención y MLP de un transformer real, no solo en MLPs.

Ejecutar con: CUDA_VISIBLE_DEVICES=0 python fase3_gpt2_flops.py
"""

import json, torch, numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch.nn.functional as F

DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED          = 42
MODEL_ID      = "gpt2"
LR            = 2e-4
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
MAX_SEQ_LEN   = 256
EPOCHS        = 2
MAX_SAMPLES   = 3000
GL_LAMBDA     = 1e-2
OMEGA_LAMBDA  = 0.05
OMEGA_EVERY_K = 10
OMEGA_PROBES  = 3
ZERO_THRESH   = 1e-3
DEAD_FRAC     = 0.95
OUT_FILE      = "results_gpt2_flops.json"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------
def load_model():
    print(f"Cargando {MODEL_ID}...")
    tok = GPT2Tokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {total:,}")
    return model, tok

# ---------------------------------------------------------------------------
# DATOS
# ---------------------------------------------------------------------------
def get_loader(tokenizer, max_s=MAX_SAMPLES):
    print("Cargando CodeAlpaca...")
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

def omega_penalty(model):
    total, count = 0.0, 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() == 2 and p.shape[0] >= 32:
            total += hutchinson_tr_a3(p.data)
            count += p.numel()
    return OMEGA_LAMBDA * total / max(count, 1)

def group_lasso_penalty(model):
    total = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() == 2 and p.shape[0] >= 32:
            total += torch.norm(p, dim=0).sum()
    return GL_LAMBDA * total

# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------
def train(model, loader, label="Omega-S+GL"):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step = 0
    for epoch in range(EPOCHS):
        total, n = 0.0, 0
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)
            mask = batch.get("attention_mask",
                              torch.ones_like(ids)).to(DEVICE)
            loss = model(input_ids=ids, attention_mask=mask,
                         labels=labs).loss / GRAD_ACCUM
            loss = loss + group_lasso_penalty(model) / GRAD_ACCUM
            if step % OMEGA_EVERY_K == 0:
                loss = loss + omega_penalty(model) / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            total += loss.item() * GRAD_ACCUM; n += 1; step += 1
            if step % 100 == 0:
                print(f"  [{label}] step {step} loss={total/n:.4f}")
        print(f"  Epoch {epoch+1}/{EPOCHS} loss={total/n:.4f}")
    return model

def train_wd(model, loader, wd_lambda=1e-2, label="WeightDecay"):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    step = 0
    for epoch in range(EPOCHS):
        total, n = 0.0, 0
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)
            mask = batch.get("attention_mask",
                              torch.ones_like(ids)).to(DEVICE)
            loss = model(input_ids=ids, attention_mask=mask,
                         labels=labs).loss / GRAD_ACCUM
            l2 = sum(p.pow(2).sum() for p in model.parameters()
                     if p.requires_grad)
            loss = loss + wd_lambda * l2 / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            total += loss.item() * GRAD_ACCUM; n += 1; step += 1
            if step % 100 == 0:
                print(f"  [{label}] step {step} loss={total/n:.4f}")
        print(f"  Epoch {epoch+1}/{EPOCHS} loss={total/n:.4f}")
    return model

# ---------------------------------------------------------------------------
# ANÁLISIS DE SPARSITY Y FLOPs
# ---------------------------------------------------------------------------
def analyze_sparsity_and_flops(model, label):
    results = {}
    total_orig, total_pruned = 0, 0
    dead_cols_total, total_cols = 0, 0

    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() == 2 and p.shape[0] >= 32:
            w = p.data.detach().cpu().numpy()
            col_zero = (np.abs(w) < ZERO_THRESH).mean(axis=0)
            dead_cols = (col_zero >= DEAD_FRAC).sum()
            alive_cols = w.shape[1] - dead_cols

            flop_orig   = 2 * w.shape[0] * w.shape[1]
            flop_pruned = 2 * w.shape[0] * alive_cols

            total_orig   += flop_orig
            total_pruned += flop_pruned
            dead_cols_total += dead_cols
            total_cols      += w.shape[1]

    flop_red   = 1.0 - total_pruned / max(total_orig, 1)
    dead_frac  = dead_cols_total / max(total_cols, 1)

    print(f"\n  [{label}]")
    print(f"  Columnas muertas: {dead_cols_total}/{total_cols} ({dead_frac:.2%})")
    print(f"  Reduccion FLOPs:  {flop_red:.2%}")

    return {"flop_reduction": flop_red, "dead_col_frac": dead_frac,
            "label": label}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("FASE 3A: FLOPs EN TRANSFORMER REAL (GPT-2)")
    print("=" * 60)

    loader = get_loader

    # Modelo A: Omega-S + Group-Lasso
    print("\n--- Entrenando con Omega-S + Group-Lasso ---")
    model_omega, tok = load_model()
    loader_omega = get_loader(tok)
    model_omega = train(model_omega, loader_omega, "Omega-S+GL")
    res_omega = analyze_sparsity_and_flops(model_omega, "Omega-S+GL")

    # Modelo B: Weight Decay
    print("\n--- Entrenando con Weight Decay ---")
    model_wd, tok = load_model()
    loader_wd = get_loader(tok)
    model_wd = train_wd(model_wd, loader_wd, label="WeightDecay")
    res_wd = analyze_sparsity_and_flops(model_wd, "WeightDecay")

    # Tabla final
    print("\n" + "=" * 60)
    print("TABLA FINAL : FLOPs EN TRANSFORMER REAL (GPT-2)")
    print("=" * 60)
    print(f"{'Metodo':<20} {'Cols muertas':>14} {'Reduccion FLOPs':>16}")
    print("-" * 52)
    for r in [res_omega, res_wd]:
        print(f"{r['label']:<20} {r['dead_col_frac']:>14.2%} "
              f"{r['flop_reduction']:>16.2%}")
    print("=" * 60)

    diff = res_omega["flop_reduction"] - res_wd["flop_reduction"]
    if res_omega["flop_reduction"] > 0.10 and diff > 0.05:
        print(f"\nCONCLUSION: Omega-S produce {diff:.1%} mas reduccion de FLOPs")
        print("que Weight Decay en un transformer real. RESULTADO PUBLICABLE.")
        print("Actualizar Seccion 4.3 del preprint con estos numeros.")
    elif res_omega["flop_reduction"] > 0.05:
        print(f"\nCONCLUSION: Señal positiva ({res_omega['flop_reduction']:.1%})")
        print("Considerar mas epocas para ampliar la señal.")
    else:
        print("\nCONCLUSION: Sparsity estructurada debil en GPT-2.")
        print("Revisar GL_LAMBDA o aumentar EPOCHS.")

    with open(OUT_FILE, "w") as f:
        json.dump({"omega": res_omega, "wd": res_wd,
                   "diff_flop": diff}, f, indent=2)
    print(f"\nResultados guardados en {OUT_FILE}")
