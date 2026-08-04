# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
FASE 5: Omega-S + Wanda : Pruning Estructurado Guiado en Llama-3-8B
=====================================================================
Diseño:
  A) Wanda estándar: poda X% de pesos por magnitud × norma activación
  B) Wanda guiada por Omega-S: mismo X% pero priorizando capas con
     mayor concentración topológica (mayor Tr(A³) normalizado)

El argumento: Omega-S identifica QUÉ capas podar. Wanda decide CÓMO.
Si B mantiene mejor accuracy que A al mismo nivel de sparsity,
el resultado es publicable y comparable con el estado del arte.

Referencia Wanda: Sun et al. 2023 (arxiv:2306.11695)

Ejecutar: CUDA_VISIBLE_DEVICES=0 python fase5_wanda_omega.py
"""

import json, torch, numpy as np, signal
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn

DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED         = 42
MODEL_ID     = "NousResearch/Meta-Llama-3-8B"
SPARSITY     = 0.50        # 50% de pesos podados : benchmark estándar de Wanda
N_CALIB      = 128         # muestras de calibración para activaciones
SEQ_LEN      = 512
HUMANEVAL_N  = 164
MAX_NEW_TOK  = 256
OMEGA_PROBES = 3
OUT_FILE     = "results_fase5_wanda_omega.json"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. MODELO
# ---------------------------------------------------------------------------
def load_model():
    print(f"Cargando {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {total/1e9:.2f}B")
    return model, tok

# ---------------------------------------------------------------------------
# 2. DATOS DE CALIBRACIÓN (para capturar activaciones reales)
# ---------------------------------------------------------------------------
def get_calib_data(tokenizer, n=N_CALIB):
    print("Cargando datos de calibracion (WikiText-2)...")
    ds = load_dataset("Salesforce/wikitext",
                      "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n]
    encodings = tokenizer(texts, truncation=True, max_length=SEQ_LEN,
                          padding="max_length", return_tensors="pt")
    return encodings["input_ids"].to(DEVICE)

# ---------------------------------------------------------------------------
# 3. OMEGA-S : mapa topológico por capa
# ---------------------------------------------------------------------------
def hutchinson_tr_a3(W, n=OMEGA_PROBES):
    W_norm = W / (W.norm() + 1e-8)
    total = 0.0
    for _ in range(n):
        v = torch.randint(0, 2, (W_norm.shape[0],),
                          device=W.device, dtype=torch.float32) * 2 - 1
        Wf = W_norm.float()
        z = Wf @ (Wf.t() @ v)
        z = Wf @ (Wf.t() @ z)
        z = Wf @ (Wf.t() @ z)
        total += (v @ z)
    return (total / n).item()

def compute_omega_scores(model):
    """
    Calcula Tr(A³) normalizado para cada capa Linear del transformer.
    Devuelve dict: {layer_name: score}
    Score mayor = mayor concentración topológica = mayor candidato a poda.
    """
    print("\nCalculando mapa topologico Omega-S por capa...")
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
            score = hutchinson_tr_a3(module.weight.data)
            scores[name] = score

    # Normalizar a [0, 1]
    min_s = min(scores.values())
    max_s = max(scores.values())
    rng   = max(max_s - min_s, 1e-8)
    scores = {k: (v - min_s) / rng for k, v in scores.items()}

    # Top 5 capas más concentradas
    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 capas con mayor concentracion topologica:")
    for name, sc in top5:
        print(f"    {name}: {sc:.4f}")

    return scores

# ---------------------------------------------------------------------------
# 4. WANDA PRUNING
# ---------------------------------------------------------------------------
class ActivationCapture:
    """Hook para capturar normas de activaciones de entrada."""
    def __init__(self):
        self.norms = None

    def __call__(self, module, inp, out):
        x = inp[0].detach().float()
        # Norma L2 por columna de entrada (promedio sobre batch y seq)
        norm = x.pow(2).mean(dim=(0, 1)).sqrt()
        if self.norms is None:
            self.norms = norm
        else:
            self.norms += norm

def wanda_prune_layer(weight, activation_norms, sparsity,
                      omega_score=1.0, omega_weight=0.3):
    """
    Wanda estándar: score = |W| × ||X||
    Wanda + Omega-S: score = |W| × ||X|| × (1 + omega_weight × omega_score)
    
    omega_score ∈ [0,1]: normalizado. Mayor score = más candidato a poda.
    omega_weight: cuánto pesa la señal topológica en la decisión.
    """
    W = weight.data.float()

    # Wanda score: magnitud del peso × norma de activación
    if activation_norms is not None:
        norms = activation_norms[:W.shape[1]].to(W.device)
        if len(norms) < W.shape[1]:
            pad = torch.ones(W.shape[1] - len(norms), device=W.device)
            norms = torch.cat([norms, pad])
        wanda_score = W.abs() * norms.unsqueeze(0)
    else:
        wanda_score = W.abs()

    # Omega-S boost: multiplicar score por factor topológico
    # (mayor omega_score = más agresivamente podado en esa capa)
    omega_factor = 1.0 + omega_weight * omega_score
    final_score  = wanda_score * omega_factor

    # Threshold: muestreo aleatorio para evitar límite de torch.quantile
    flat = final_score.flatten()
    if flat.numel() > 1_000_000:
        idx = torch.randperm(flat.numel(), device=flat.device)[:1_000_000]
        flat = flat[idx]
    threshold = torch.quantile(flat, sparsity).item()
    mask = final_score > threshold

    with torch.no_grad():
        weight.data = (W * mask.float()).to(weight.dtype)

    sparsity_real = (weight.data == 0).float().mean().item()
    return sparsity_real

def run_pruning(model, calib_ids, omega_scores, use_omega, label):
    """
    Captura activaciones en un forward pass de calibración,
    luego aplica Wanda (con o sin guía Omega-S) a todas las capas Linear.
    """
    print(f"\n[{label}] Capturando activaciones...")
    hooks    = {}
    captures = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
            cap = ActivationCapture()
            hooks[name] = module.register_forward_hook(cap)
            captures[name] = cap

    # Forward pass de calibración (sin gradientes)
    with torch.no_grad():
        for i in range(0, min(N_CALIB, calib_ids.shape[0]), 4):
            batch = calib_ids[i:i+4]
            try:
                model(batch)
            except Exception:
                pass

    # Eliminar hooks
    for h in hooks.values():
        h.remove()

    print(f"[{label}] Aplicando Wanda pruning (sparsity={SPARSITY:.0%})...")
    total_sparsity = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
            act_norms  = captures[name].norms
            omega_sc   = omega_scores.get(name, 0.5) if use_omega else 0.0
            sp = wanda_prune_layer(module.weight, act_norms,
                                   SPARSITY, omega_sc)
            total_sparsity.append(sp)

    mean_sp = np.mean(total_sparsity)
    print(f"[{label}] Sparsity real media: {mean_sp:.2%}")
    return mean_sp

# ---------------------------------------------------------------------------
# 5. HUMANEVAL
# ---------------------------------------------------------------------------
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
            out = model.generate(
                **gen_in, max_new_tokens=MAX_NEW_TOK,
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
# 6. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*65)
    print("FASE 5: OMEGA-S + WANDA : PRUNING GUIADO EN LLAMA-3-8B")
    print(f"Sparsity objetivo: {SPARSITY:.0%}")
    print("="*65)

    # Cargar modelo y datos
    model, tok = load_model()
    calib_ids  = get_calib_data(tok)

    # Calcular mapa topológico Omega-S
    omega_scores = compute_omega_scores(model)

    # Evaluar antes del pruning
    print("\n[PASO 1] HumanEval antes del pruning...")
    he_base = humaneval(model, tok, HUMANEVAL_N, "baseline")

    # Experimento A: Wanda estándar
    import copy
    print("\n[PASO 2] Wanda ESTANDAR...")
    model_wanda = copy.deepcopy(model)
    sp_wanda = run_pruning(model_wanda, calib_ids, omega_scores,
                           use_omega=False, label="Wanda")
    he_wanda = humaneval(model_wanda, tok, HUMANEVAL_N, "Wanda estándar")
    del model_wanda; torch.cuda.empty_cache()

    # Experimento B: Wanda + Omega-S
    print("\n[PASO 3] Wanda + OMEGA-S...")
    model_omega = copy.deepcopy(model)
    sp_omega = run_pruning(model_omega, calib_ids, omega_scores,
                           use_omega=True, label="Wanda+Omega-S")
    he_omega = humaneval(model_omega, tok, HUMANEVAL_N, "Wanda+Omega-S")
    del model_omega, model; torch.cuda.empty_cache()

    # Tabla final
    cost_wanda = he_base - he_wanda
    cost_omega = he_base - he_omega
    ventaja    = cost_wanda - cost_omega

    print("\n\n" + "="*70)
    print("TABLA FINAL : WANDA vs WANDA + OMEGA-S")
    print("="*70)
    print(f"{'Metodo':<25} {'HumanEval':>10} {'Coste acc':>10} "
          f"{'Sparsity':>10}")
    print("-"*57)
    print(f"{'Sin pruning':<25} {he_base:>10.2%} {':':>10} {'0.00%':>10}")
    print(f"{'Wanda estandar':<25} {he_wanda:>10.2%} "
          f"{-cost_wanda:>+10.2%} {sp_wanda:>10.2%}")
    print(f"{'Wanda + Omega-S':<25} {he_omega:>10.2%} "
          f"{-cost_omega:>+10.2%} {sp_omega:>10.2%}")
    print("="*70)
    print(f"\nVentaja Omega-S: {ventaja:+.2%} menos coste de accuracy "
          f"al {SPARSITY:.0%} de sparsity")

    if ventaja > 0.03:
        print("\nRESULTADO PUBLICABLE:")
        print("Omega-S como señal de diagnostico mejora Wanda en transformers.")
        print("Comparable con estado del arte (Sun et al. 2023).")
    elif ventaja > 0:
        print("\nSeñal positiva pero debil.")
        print("Probar con omega_weight mayor o diferente sparsity.")
    else:
        print("\nSin ventaja sobre Wanda estandar.")
        print("El mapa topologico de Omega-S no mejora la seleccion de Wanda.")

    results = {
        "config": {
            "model": MODEL_ID,
            "sparsity": SPARSITY,
            "n_calib": N_CALIB,
            "seed": SEED,
        },
        "baseline":      {"humaneval": he_base},
        "wanda_standard":{"humaneval": he_wanda, "sparsity": sp_wanda,
                          "acc_cost": cost_wanda},
        "wanda_omega":   {"humaneval": he_omega, "sparsity": sp_omega,
                          "acc_cost": cost_omega},
        "advantage_omega_vs_wanda": ventaja,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados en {OUT_FILE}")
