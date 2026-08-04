# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
TEST 3 (extensión del hilo): Omega-S + group-sparsity combinados
=============================================================================
Objetivo: el test anterior mostró que Omega-S por sí solo produce sparsity
DISPERSA (0% de filas/columnas muertas). Aquí añadimos una penalización de
group-lasso explícita (por columna = por neurona de entrada) junto a Omega-S,
para ver si la combinación sí logra agrupar los ceros en columnas completas
-> sparsity estructurada -> podable de verdad.

Se entrena sobre Split-MNIST (Tarea A: dígitos 0-4) para mantener
comparabilidad con vuestros Test 2/4/5 anteriores.

Requiere: torch, torchvision (ambos vienen preinstalados en Colab)
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3

OMEGA_LAMBDA = 0.05        # peso de la penalización topológica (Tr(A^3))
OMEGA_EVERY_K = 10         # frecuencia de aplicación de Omega (pasos)
OMEGA_N_PROBES = 4         # nº de vectores de Hutchinson por evaluación

GROUP_LASSO_LAMBDA = 1e-4  # peso del group-lasso (se aplica cada step, es barato)

ZERO_THRESHOLD = 1e-3
STRUCTURED_DEAD_FRACTION = 0.95

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. DATOS: Split-MNIST, Tarea A = dígitos 0-4
# ---------------------------------------------------------------------------
def get_task_a_loader():
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Lambda(lambda x: x.view(-1))])
    full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    idx = [i for i, (_, y) in enumerate(full) if y < 5]
    subset = Subset(full, idx)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------------------------------------------------------
# 2. MODELO
# ---------------------------------------------------------------------------
def build_model():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(DEVICE)


def linear_layers(model):
    return [l for l in model if isinstance(l, nn.Linear)]


# ---------------------------------------------------------------------------
# 3. OMEGA-S: estimador de Hutchinson para Tr((W W^T)^3)
# ---------------------------------------------------------------------------
def hutchinson_tr_a3(W, n_probes=OMEGA_N_PROBES):
    """
    Estima Tr(A^3) donde A = W W^T, sin materializar A.
    Para cada vector aleatorio v (Rademacher):
        z1 = A v  = W (W^T v)
        z2 = A z1 = W (W^T z1)
        z3 = A z2 = W (W^T z2)
        estimador_i = v^T z3
    Tr(A^3) ~= media de estimador_i sobre n_probes muestras.
    """
    m, n = W.shape  # W: (out_features, in_features) -> A = W W^T es (out, out)
    total = 0.0
    for _ in range(n_probes):
        v = torch.randint(0, 2, (m,), device=W.device, dtype=W.dtype) * 2 - 1  # Rademacher +-1
        z = W @ (W.t() @ v)
        z = W @ (W.t() @ z)
        z = W @ (W.t() @ z)
        total = total + (v @ z)
    return total / n_probes


def omega_penalty(model, n_probes=OMEGA_N_PROBES):
    penalty = 0.0
    for layer in linear_layers(model):
        penalty = penalty + hutchinson_tr_a3(layer.weight, n_probes)
    return penalty


# ---------------------------------------------------------------------------
# 4. GROUP-LASSO: penaliza columnas completas (neuronas de entrada) a la vez
# ---------------------------------------------------------------------------
def group_lasso_penalty(model):
    """
    Suma, para cada capa, la norma L2 de cada columna de pesos.
    Minimizar esto empuja a que columnas ENTERAS colapsen a cero -- eso es
    lo que test1 no encontró y lo que este término intenta forzar.
    """
    penalty = 0.0
    for layer in linear_layers(model):
        col_norms = torch.norm(layer.weight, dim=0)  # una norma por columna de entrada
        penalty = penalty + col_norms.sum()
    return penalty


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO
# ---------------------------------------------------------------------------
def train(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    step = 0
    for epoch in range(EPOCHS):
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            logits = model(x)
            task_loss = F.cross_entropy(logits, y)

            loss = task_loss + GROUP_LASSO_LAMBDA * group_lasso_penalty(model)

            if step % OMEGA_EVERY_K == 0:
                loss = loss + OMEGA_LAMBDA * omega_penalty(model)

            loss.backward()
            optimizer.step()

            total_loss += task_loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
            step += 1

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/n:.4f} - "
              f"accuracy: {correct/n:.4%}")
    return model


# ---------------------------------------------------------------------------
# 6. ANÁLISIS DE ESTRUCTURA (mismo criterio que test1_sparsity_estructural.py)
# ---------------------------------------------------------------------------
def analyze_layer_sparsity(weight, name):
    w = weight.detach().cpu().numpy()
    zero_mask = np.abs(w) < ZERO_THRESHOLD
    total_sparsity = zero_mask.mean()
    row_zero_frac = zero_mask.mean(axis=1)
    col_zero_frac = zero_mask.mean(axis=0)
    dead_rows = (row_zero_frac >= STRUCTURED_DEAD_FRACTION).sum()
    dead_cols = (col_zero_frac >= STRUCTURED_DEAD_FRACTION).sum()

    print(f"\n--- Capa: {name} ---")
    print(f"  Forma: {w.shape}")
    print(f"  Sparsity total: {total_sparsity:.4%}")
    print(f"  Filas ~muertas: {dead_rows}/{w.shape[0]} ({dead_rows/w.shape[0]:.2%})")
    print(f"  Columnas ~muertas: {dead_cols}/{w.shape[1]} ({dead_cols/w.shape[1]:.2%})")

    is_structured = (dead_rows / w.shape[0] > 0.02) or (dead_cols / w.shape[1] > 0.02)
    print(f"  Veredicto: {'ESTRUCTURADA (podable)' if is_structured else 'DISPERSA'}")
    return is_structured


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Dispositivo: {DEVICE}")
    loader = get_task_a_loader()
    model = build_model()

    print("=" * 70)
    print("ENTRENANDO CON OMEGA-S + GROUP-LASSO COMBINADOS")
    print("=" * 70)
    model = train(model, loader)

    print("\n" + "=" * 70)
    print("ANÁLISIS DE ESTRUCTURA DE SPARSITY (post-entrenamiento)")
    print("=" * 70)
    any_structured = False
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            structured = analyze_layer_sparsity(layer.weight, f"Linear_{i}")
            any_structured = any_structured or structured

    print("\n" + "=" * 70)
    if any_structured:
        print("CONCLUSIÓN: la combinación SÍ produce sparsity estructurada.")
        print("-> Reutiliza prune_structured() de test1_sparsity_estructural.py")
        print("   para medir la reducción real de FLOPs sobre este checkpoint.")
    else:
        print("CONCLUSIÓN: sigue sin ser estructurada incluso con group-lasso.")
        print("-> Prueba a subir GROUP_LASSO_LAMBDA (p.ej. 1e-3) o más épocas;")
        print("   si persiste, la vía de 'ahorro de energía' necesita un enfoque")
        print("   distinto (p.ej. pruning explícito guiado por Omega en vez de")
        print("   esperar que emerja solo del entrenamiento).")
    print("=" * 70)
