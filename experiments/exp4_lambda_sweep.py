# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
TEST 4: Barrido automático de GROUP_LASSO_LAMBDA (Omega-S + Group-Lasso)
=============================================================================
Entrena un MLP con Omega-S + group-lasso para cada valor de lambda en
LAMBDA_GRID y reporta una tabla comparativa al final:

  lambda | accuracy | sparsity_total | cols_muertas | filas_muertas | FLOPs_reduccion

Así ves en una sola corrida en qué punto aparece sparsity estructurada
y cuánto accuracy cuesta : en vez de probar valores a mano uno a uno.

Requiere: torch, torchvision (preinstalados en Colab)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# ---------------------------------------------------------------------------
# 0. CONFIGURACIÓN
# ---------------------------------------------------------------------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42
EPOCHS        = 5
BATCH_SIZE    = 128
LR            = 1e-3

OMEGA_LAMBDA  = 0.05
OMEGA_EVERY_K = 10
OMEGA_N_PROBES = 4

# Barrido: de muy suave a agresivo, logarítmico
LAMBDA_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

ZERO_THRESHOLD          = 1e-3
STRUCTURED_DEAD_FRACTION = 0.95

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
def get_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    idx  = [i for i, (_, y) in enumerate(full) if y < 5]
    return DataLoader(Subset(full, idx), batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------------------------------------------------------
# 2. MODELO : se reconstruye limpio para cada lambda del barrido
# ---------------------------------------------------------------------------
def build_model():
    return nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128,  10),
    ).to(DEVICE)


def linear_layers(model):
    return [l for l in model if isinstance(l, nn.Linear)]


# ---------------------------------------------------------------------------
# 3. OMEGA-S (mismo estimador que test3)
# ---------------------------------------------------------------------------
def hutchinson_tr_a3(W, n_probes=OMEGA_N_PROBES):
    total = 0.0
    for _ in range(n_probes):
        v = torch.randint(0, 2, (W.shape[0],), device=W.device, dtype=W.dtype) * 2 - 1
        z = W @ (W.t() @ v)
        z = W @ (W.t() @ z)
        z = W @ (W.t() @ z)
        total = total + (v @ z)
    return total / n_probes


def omega_penalty(model):
    return sum(hutchinson_tr_a3(l.weight) for l in linear_layers(model))


# ---------------------------------------------------------------------------
# 4. GROUP-LASSO (norma L2 por columna)
# ---------------------------------------------------------------------------
def group_lasso_penalty(model):
    return sum(torch.norm(l.weight, dim=0).sum() for l in linear_layers(model))


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO
# ---------------------------------------------------------------------------
def train(model, loader, gl_lambda):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    step = 0
    for epoch in range(EPOCHS):
        correct, n = 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            if gl_lambda > 0:
                loss = loss + gl_lambda * group_lasso_penalty(model)
            if step % OMEGA_EVERY_K == 0:
                loss = loss + OMEGA_LAMBDA * omega_penalty(model)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            n       += x.size(0)
            step    += 1
        print(f"  epoch {epoch+1}/{EPOCHS}  acc={correct/n:.4%}")
    return model


# ---------------------------------------------------------------------------
# 6. ANÁLISIS DE SPARSITY
# ---------------------------------------------------------------------------
def analyze_sparsity(model):
    """
    Devuelve un dict con:
    - sparsity_total    : fracción de pesos ~0 en todo el modelo
    - dead_col_frac     : fracción de columnas ~muertas (promedio entre capas)
    - dead_row_frac     : fracción de filas ~muertas
    - any_structured    : bool
    - flop_reduction    : estimación de reducción de FLOPs si se podaran esas cols/filas
    """
    total_elems   = 0
    total_zeros   = 0
    dead_cols_all = 0
    total_cols    = 0
    dead_rows_all = 0
    total_rows    = 0
    flop_orig     = 0
    flop_pruned   = 0

    layers = linear_layers(model)
    prev_dead_rows = None  # columnas a podar en capa siguiente = filas muertas de la anterior

    for i, layer in enumerate(layers):
        w = layer.weight.detach().cpu().numpy()
        zero_mask    = np.abs(w) < ZERO_THRESHOLD
        total_elems += w.size
        total_zeros += zero_mask.sum()

        row_zero_frac = zero_mask.mean(axis=1)
        col_zero_frac = zero_mask.mean(axis=0)
        dead_rows = (row_zero_frac >= STRUCTURED_DEAD_FRACTION).sum()
        dead_cols = (col_zero_frac >= STRUCTURED_DEAD_FRACTION).sum()

        dead_rows_all += dead_rows
        dead_cols_all += dead_cols
        total_rows    += w.shape[0]
        total_cols    += w.shape[1]

        # FLOPs originales (2*in*out)
        flop_orig   += 2 * w.shape[1] * w.shape[0]
        # FLOPs tras podar cols muertas de esta capa Y filas muertas propagadas
        in_keep  = w.shape[1] - dead_cols
        out_keep = w.shape[0] - dead_rows
        flop_pruned += 2 * in_keep * out_keep

    sparsity_total = total_zeros / total_elems
    dead_col_frac  = dead_cols_all / total_cols
    dead_row_frac  = dead_rows_all / total_rows
    any_structured = (dead_col_frac > 0.02) or (dead_row_frac > 0.02)
    flop_reduction = 1.0 - (flop_pruned / flop_orig)

    return {
        "sparsity_total": sparsity_total,
        "dead_col_frac":  dead_col_frac,
        "dead_row_frac":  dead_row_frac,
        "any_structured": any_structured,
        "flop_reduction": flop_reduction,
    }


def eval_accuracy(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            n       += x.size(0)
    model.train()
    return correct / n


# ---------------------------------------------------------------------------
# 7. MAIN : barrido completo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Dispositivo: {DEVICE}")
    loader = get_loader()

    resultados = []

    for gl_lambda in LAMBDA_GRID:
        label = f"λ={gl_lambda:.0e}" if gl_lambda > 0 else "λ=0.0 (solo Omega-S)"
        print("\n" + "=" * 70)
        print(f"ENTRENANDO: {label}")
        print("=" * 70)
        model = build_model()
        torch.manual_seed(SEED)
        model = train(model, loader, gl_lambda)

        acc     = eval_accuracy(model, loader)
        sp      = analyze_sparsity(model)

        resultados.append({
            "lambda":         gl_lambda,
            "accuracy":       acc,
            **sp,
        })
        print(f"  -> acc={acc:.4%} | sparsity={sp['sparsity_total']:.2%} | "
              f"cols_muertas={sp['dead_col_frac']:.2%} | "
              f"filas_muertas={sp['dead_row_frac']:.2%} | "
              f"reducción_FLOPs≈{sp['flop_reduction']:.2%} | "
              f"{'✓ ESTRUCTURADA' if sp['any_structured'] else '✗ dispersa'}")

    # -----------------------------------------------------------------------
    # TABLA RESUMEN
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("TABLA RESUMEN DEL BARRIDO")
    print("=" * 90)
    print(f"{'λ group-lasso':<20} {'Accuracy':>10} {'Sparsity':>10} "
          f"{'Cols~0':>10} {'Filas~0':>10} {'ΔFLOP':>10} {'Estructura':>12}")
    print("-" * 90)
    for r in resultados:
        lbl = f"{r['lambda']:.0e}" if r['lambda'] > 0 else "0.0 (baseline)"
        print(f"{lbl:<20} {r['accuracy']:>10.4%} {r['sparsity_total']:>10.2%} "
              f"{r['dead_col_frac']:>10.2%} {r['dead_row_frac']:>10.2%} "
              f"{r['flop_reduction']:>10.2%} "
              f"{'✓ PODABLE' if r['any_structured'] else '✗ dispersa':>12}")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # CONCLUSIÓN AUTOMÁTICA
    # -----------------------------------------------------------------------
    structured = [r for r in resultados if r["any_structured"]]
    if not structured:
        print("\nCONCLUSIÓN: ningún valor de λ produjo sparsity estructurada.")
        print("La vía 'Omega-S -> energía vía pruning' necesita un enfoque distinto.")
        print("Recomendación: pruning explícito post-entrenamiento guiado por Omega,")
        print("en vez de esperar que la estructura emerja sola del entrenamiento.")
    else:
        # El mejor es el que tiene más reducción de FLOPs sin caer >2pp de accuracy
        baseline_acc = resultados[0]["accuracy"]
        viable = [r for r in structured if baseline_acc - r["accuracy"] < 0.02]
        if viable:
            best = max(viable, key=lambda r: r["flop_reduction"])
            print(f"\nCONCLUSIÓN: λ={best['lambda']:.0e} es el punto óptimo.")
            print(f"  Accuracy: {best['accuracy']:.4%} (caída de "
                  f"{(baseline_acc - best['accuracy'])*100:.2f}pp frente a baseline)")
            print(f"  Reducción estimada de FLOPs: {best['flop_reduction']:.2%}")
            print(f"  -> Vale la pena continuar esta línea en un modelo de lenguaje.")
        else:
            best = max(structured, key=lambda r: r["flop_reduction"])
            print(f"\nCONCLUSIÓN: hay sparsity estructurada desde λ={best['lambda']:.0e}")
            print(f"  pero con un coste de accuracy de "
                  f"{(baseline_acc - best['accuracy'])*100:.2f}pp : demasiado alto.")
            print("  Prueba a aumentar EPOCHS o reducir OMEGA_LAMBDA antes de escalar.")
    print("=" * 90)
