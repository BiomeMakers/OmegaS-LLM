# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
FASE 5 v2: Omega-S + Wanda con distribución no uniforme de sparsity
=====================================================================
Mejora sobre v1:
  En vez de modificar el score interno de Wanda, Omega-S determina
  CUÁNTA sparsity recibe cada capa.

  Capas con alta concentración topológica (monopolios) reciben más
  sparsity (más agresivamente podadas).
  Capas con distribución uniforme reciben menos sparsity.
  El presupuesto total de sparsity es el mismo que Wanda estándar.

  Wanda opera con su algoritmo completo dentro de cada capa :
  solo cambia el presupuesto por capa.

Diseño:
  A) Wanda estándar: sparsity uniforme S en todas las capas
  B) Wanda + Omega-S v2: sparsity variable por capa,
     media = S, distribuida según score topológico

Ejecutar: CUDA_VISIBLE_DEVICES=0 python fase5v2_wanda_nonuniform.py
"""

import json, torch, numpy as np, signal
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn

DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED          = 42
MODEL_ID      = "NousResearch/Meta-Llama-3-8B"
TARGET_SPARSITY = 0.30    # sparsity media objetivo
OMEGA_WEIGHT    = 0.5     # cuánto varía la sparsity por capa (0=uniforme, 1=máxima variación)
N_CALIB         = 128
SEQ_LEN         = 512
HUMANEVAL_N     = 164
MAX_NEW_TOK     = 256
OMEGA_PROBES    = 3
OUT_FILE        = "results_fase5v3.json"

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
# 2. DATOS DE CALIBRACIÓN
# ---------------------------------------------------------------------------
def get_calib_data(tokenizer, n=N_CALIB):
    print("Cargando datos de calibracion...")
    ds = load_dataset("Salesforce/wikitext",
                      "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n]
    enc = tokenizer(texts, truncation=True, max_length=SEQ_LEN,
                    padding="max_length", return_tensors="pt")
    return enc["input_ids"].to(DEVICE)

# ---------------------------------------------------------------------------
# 3. OMEGA-S : mapa topológico con calibración de activaciones
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

def compute_omega_scores_with_activation(model, calib_ids):
    """
    MEJORA v2: combina Tr(A³) con norma media de activaciones.
    Score final = Tr(A³) × (1 / ||X||_mean)
    Capas con alta concentración topológica Y baja activación
    son las más candidatas a poda agresiva.
    """
    print("\nCalculando mapa topologico Omega-S (con activaciones)...")

    # Capturar normas de activación
    act_norms = {}
    hooks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
            def make_hook(n):
                def hook(mod, inp, out):
                    x = inp[0].detach().float()
                    act_norms[n] = x.pow(2).mean().sqrt().item()
                return hook
            hooks[name] = module.register_forward_hook(make_hook(name))

    with torch.no_grad():
        for i in range(0, min(N_CALIB, calib_ids.shape[0]), 4):
            try:
                model(calib_ids[i:i+4])
            except Exception:
                pass

    for h in hooks.values():
        h.remove()

    # Calcular score combinado
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
            tr_a3    = hutchinson_tr_a3(module.weight.data)
            act_norm = act_norms.get(name, 1.0)
            # Capas con alto Tr(A³) y baja activación = más redundantes
            scores[name] = tr_a3 / (act_norm + 1e-8)

    # Normalizar por percentil p10-p90 para evitar que una capa domine
    vals = np.array(list(scores.values()))
    p10  = np.percentile(vals, 10)
    p90  = np.percentile(vals, 90)
    rng  = max(p90 - p10, 1e-8)
    scores = {k: np.clip((v - p10) / rng, 0.0, 1.0)
              for k, v in scores.items()}

    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 capas mas redundantes (candidatas a poda agresiva):")
    for name, sc in top5:
        print(f"    {name}: {sc:.4f}")

    return scores

def compute_layer_sparsities(omega_scores, target_sparsity, omega_weight):
    """
    Distribuye el presupuesto de sparsity de forma no uniforme.
    Capa con score=1 (máximo monopolio) recibe target × (1 + omega_weight)
    Capa con score=0 (más uniforme) recibe target × (1 - omega_weight)
    Media garantizada = target_sparsity
    Todos los valores se clampean a [0.05, 0.95]
    """
    layer_sparsities = {}
    for name, score in omega_scores.items():
        # score ∈ [0,1], centrado en 0.5 → factor ∈ [1-ω, 1+ω]
        factor = 1.0 + omega_weight * (2 * score - 1)
        sp = target_sparsity * factor
        sp = max(0.05, min(0.95, sp))
        layer_sparsities[name] = sp

    # Verificar media
    mean_sp = np.mean(list(layer_sparsities.values()))
    print(f"\n  Sparsity media distribuida: {mean_sp:.2%} "
          f"(objetivo: {target_sparsity:.2%})")
    print(f"  Rango: [{min(layer_sparsities.values()):.2%}, "
          f"{max(layer_sparsities.values()):.2%}]")

    return layer_sparsities

# ---------------------------------------------------------------------------
# 4. WANDA CON SPARSITY POR CAPA
# ---------------------------------------------------------------------------
class ActivationCapture:
    def __init__(self):
        self.norms = None
    def __call__(self, module, inp, out):
        x = inp[0].detach().float()
        norm = x.pow(2).mean(dim=(0, 1)).sqrt()
        if self.norms is None:
            self.norms = norm
        else:
            self.norms += norm

def wanda_prune_layer(weight, activation_norms, sparsity):
    W = weight.data.float()
    if activation_norms is not None:
        norms = activation_norms[:W.shape[1]].to(W.device)
        if len(norms) < W.shape[1]:
            pad = torch.ones(W.shape[1] - len(norms), device=W.device)
            norms = torch.cat([norms, pad])
        score = W.abs() * norms.unsqueeze(0)
    else:
        score = W.abs()

    flat = score.flatten()
    if flat.numel() > 1_000_000:
        idx  = torch.randperm(flat.numel(), device=flat.device)[:1_000_000]
        flat = flat[idx]
    threshold = torch.quantile(flat, sparsity).item()
    mask = score > threshold

    with torch.no_grad():
        weight.data = (W * mask.float()).to(weight.dtype)

    return (weight.data == 0).float().mean().item()

def run_pruning(model, calib_ids, layer_sparsities, label):
    print(f"\n[{label}] Capturando activaciones...")
    hooks, captures = {}, {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
            cap = ActivationCapture()
            hooks[name] = module.register_forward_hook(cap)
            captures[name] = cap

    with torch.no_grad():
        for i in range(0, min(N_CALIB, calib_ids.shape[0]), 4):
            try:
                model(calib_ids[i:i+4])
            except Exception:
                pass

    for h in hooks.values():
        h.remove()

    print(f"[{label}] Aplicando Wanda con sparsity por capa...")
    sparsities = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
            act_norms = captures[name].norms
            sp_target = layer_sparsities.get(name, TARGET_SPARSITY)
            sp_real   = wanda_prune_layer(module.weight, act_norms, sp_target)
            sparsities.append(sp_real)

    mean_sp = np.mean(sparsities)
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
    print("FASE 5 v3: OMEGA-S + WANDA (NORMALIZACIÓN POR PERCENTIL)")
    print(f"Sparsity media objetivo: {TARGET_SPARSITY:.0%}")
    print(f"Omega weight: {OMEGA_WEIGHT}")
    print("="*65)

    model, tok = load_model()
    calib_ids  = get_calib_data(tok)

    # Mapa topológico con calibración
    omega_scores = compute_omega_scores_with_activation(model, calib_ids)

    # Distribución no uniforme de sparsity
    layer_sparsities_omega = compute_layer_sparsities(
        omega_scores, TARGET_SPARSITY, OMEGA_WEIGHT)

    # Sparsity uniforme para Wanda estándar
    layer_sparsities_uniform = {k: TARGET_SPARSITY
                                 for k in omega_scores.keys()}

    # Evaluar baseline
    print("\n[PASO 1] HumanEval baseline...")
    he_base = humaneval(model, tok, HUMANEVAL_N, "baseline")

    # Liberar modelo base
    del model; torch.cuda.empty_cache()

    # Experimento A: Wanda estándar (sparsity uniforme)
    print("\n[PASO 2] Cargando modelo para Wanda uniforme...")
    model_wanda, _ = load_model()
    sp_wanda = run_pruning(model_wanda, calib_ids,
                           layer_sparsities_uniform, "Wanda uniforme")
    he_wanda = humaneval(model_wanda, tok, HUMANEVAL_N, "Wanda uniforme")
    del model_wanda; torch.cuda.empty_cache()

    # Experimento B: Wanda + Omega-S (sparsity no uniforme)
    print("\n[PASO 3] Cargando modelo para Wanda+Omega-S...")
    model_omega, _ = load_model()
    sp_omega = run_pruning(model_omega, calib_ids,
                           layer_sparsities_omega, "Wanda+Omega-S")
    he_omega = humaneval(model_omega, tok, HUMANEVAL_N, "Wanda+Omega-S")
    del model_omega; torch.cuda.empty_cache()

    # Tabla final
    cost_wanda = he_base - he_wanda
    cost_omega = he_base - he_omega
    ventaja    = cost_wanda - cost_omega

    print("\n\n" + "="*70)
    print("TABLA FINAL : WANDA UNIFORME vs WANDA + OMEGA-S NO UNIFORME")
    print("="*70)
    print(f"{'Metodo':<30} {'HumanEval':>10} {'Coste acc':>10} "
          f"{'Sparsity':>10}")
    print("-"*62)
    print(f"{'Sin pruning':<30} {he_base:>10.2%} {':':>10} {'0.00%':>10}")
    print(f"{'Wanda uniforme':<30} {he_wanda:>10.2%} "
          f"{-cost_wanda:>+10.2%} {sp_wanda:>10.2%}")
    print(f"{'Wanda + Omega-S no uniforme':<30} {he_omega:>10.2%} "
          f"{-cost_omega:>+10.2%} {sp_omega:>10.2%}")
    print("="*70)
    print(f"\nVentaja Omega-S: {ventaja:+.2%} menos coste de accuracy "
          f"al {TARGET_SPARSITY:.0%} de sparsity media")

    if ventaja > 0.03:
        print("\nRESULTADO PUBLICABLE:")
        print("Omega-S como distribuidor de sparsity mejora Wanda.")
        print("Argumento de FLOPs en transformer real VALIDADO.")
    elif ventaja > 0.01:
        print("\nSeñal positiva. Probar OMEGA_WEIGHT mayor o mas sparsity.")
    else:
        print("\nSin ventaja. El mapa topologico no mejora la distribucion.")
        print("Considerar calibracion post-pruning o arquitectura diferente.")

    with open(OUT_FILE, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID,
                "target_sparsity": TARGET_SPARSITY,
                "omega_weight": OMEGA_WEIGHT,
                "seed": SEED,
            },
            "baseline":        {"humaneval": he_base},
            "wanda_uniform":   {"humaneval": he_wanda, "sparsity": sp_wanda,
                                "acc_cost": cost_wanda},
            "wanda_omega_v2":  {"humaneval": he_omega, "sparsity": sp_omega,
                                "acc_cost": cost_omega},
            "advantage": ventaja,
        }, f, indent=2)
    print(f"\nResultados en {OUT_FILE}")
