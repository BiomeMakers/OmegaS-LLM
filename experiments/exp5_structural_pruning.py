# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
TEST 5 (corregido): Validación de poda estructurada real sobre el checkpoint λ=1e-2
=============================================================================
Corrección respecto a la versión anterior: la propagación ahora va en ambas
direcciones. Si capa i+1 elimina columnas (neuronas de entrada), esas mismas
neuronas son filas de salida de capa i y deben eliminarse allí también.
El algoritmo ahora hace dos pasadas:
  1. Hacia adelante: detecta columnas muertas por layer y propaga filas muertas.
  2. Hacia atrás:   propaga hacia la capa anterior cualquier columna eliminada
                    que no estuviera ya muerta en esa capa (reconciliación).
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
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
SEED               = 42
EPOCHS             = 5
BATCH_SIZE         = 128
LR                 = 1e-3
OMEGA_LAMBDA       = 0.05
OMEGA_EVERY_K      = 10
OMEGA_N_PROBES     = 4
GL_LAMBDA          = 1e-2
ZERO_THRESHOLD     = 1e-3
DEAD_FRACTION      = 0.95

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
def get_loader(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    full = datasets.MNIST(root="./data", train=train, download=True, transform=transform)
    idx  = [i for i, (_, y) in enumerate(full) if y < 5]
    return DataLoader(Subset(full, idx), batch_size=BATCH_SIZE,
                      shuffle=train, drop_last=False)


# ---------------------------------------------------------------------------
# 2. MODELO
# ---------------------------------------------------------------------------
def build_model():
    return nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128,  10),
    ).to(DEVICE)


def get_linear_layers(model):
    """Devuelve lista de (indice_en_sequential, capa_linear) en orden."""
    return [(i, l) for i, l in enumerate(model) if isinstance(l, nn.Linear)]


# ---------------------------------------------------------------------------
# 3. OMEGA-S + GROUP-LASSO
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
    return sum(hutchinson_tr_a3(l.weight) for _, l in get_linear_layers(model))

def group_lasso_penalty(model):
    return sum(torch.norm(l.weight, dim=0).sum() for _, l in get_linear_layers(model))


# ---------------------------------------------------------------------------
# 4. ENTRENAMIENTO
# ---------------------------------------------------------------------------
def train(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    step = 0
    for epoch in range(EPOCHS):
        correct, n = 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            loss   = loss + GL_LAMBDA * group_lasso_penalty(model)
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
# 5. DETECCIÓN DE NEURONAS MUERTAS
# ---------------------------------------------------------------------------
def dead_col_indices(w_np):
    """Columnas donde >= DEAD_FRACTION de pesos son ~0."""
    return np.where((np.abs(w_np) < ZERO_THRESHOLD).mean(axis=0) >= DEAD_FRACTION)[0]

def dead_row_indices(w_np):
    """Filas donde >= DEAD_FRACTION de pesos son ~0."""
    return np.where((np.abs(w_np) < ZERO_THRESHOLD).mean(axis=1) >= DEAD_FRACTION)[0]


# ---------------------------------------------------------------------------
# 6. PODA CON PROPAGACIÓN BIDIRECCIONAL
# ---------------------------------------------------------------------------
def compute_keep_indices(model):
    """
    Calcula qué índices de entrada/salida conservar en cada capa lineal,
    propagando la poda en ambas direcciones para garantizar consistencia
    de dimensiones entre capas adyacentes.

    Retorna lista de dicts: [{keep_in, keep_out, seq_idx}, ...]
    """
    ll = get_linear_layers(model)  # [(seq_idx, layer), ...]
    n  = len(ll)

    weights = [l.weight.detach().cpu().numpy() for _, l in ll]

    # Paso 1 : detectar muertes locales en cada capa
    local_dead_cols = [dead_col_indices(w) for w in weights]
    local_dead_rows = [dead_row_indices(w) for w in weights]

    # keep_out[i] = neuronas de salida activas de la capa i
    # keep_in[i]  = neuronas de entrada activas de la capa i
    keep_out = [np.setdiff1d(np.arange(w.shape[0]), local_dead_rows[i])
                for i, w in enumerate(weights)]
    keep_in  = [np.setdiff1d(np.arange(w.shape[1]), local_dead_cols[i])
                for i, w in enumerate(weights)]

    # Paso 2 : reconciliar: keep_in[i+1] debe ser subconjunto de keep_out[i]
    # Si capa i+1 eliminó columnas que capa i todavía produce, eliminarlas de keep_out[i]
    for i in range(n - 1):
        # neuronas que capa i produce pero capa i+1 no consume
        unused = np.setdiff1d(keep_out[i], keep_in[i + 1])
        keep_out[i] = np.intersect1d(keep_out[i], keep_in[i + 1])
        # propagar hacia atrás: si keep_out[i] se redujo, actualizar keep_in[i+1]
        keep_in[i + 1] = keep_out[i].copy()

    # La capa de salida (última) nunca poda filas (no eliminar clases)
    keep_out[-1] = np.arange(weights[-1].shape[0])

    return [{"seq_idx": ll[i][0],
             "keep_in":  keep_in[i],
             "keep_out": keep_out[i],
             "shape_orig": weights[i].shape}
            for i in range(n)]


def build_pruned_model(model, keep_info):
    """Construye un nuevo Sequential con los pesos podados."""
    layer_map = {info["seq_idx"]: info for info in keep_info}
    new_mods  = []
    for seq_idx, module in enumerate(model):
        if isinstance(module, nn.Linear):
            info     = layer_map[seq_idx]
            ki, ko   = info["keep_in"], info["keep_out"]
            w        = module.weight.detach().cpu().numpy()[np.ix_(ko, ki)]
            b        = module.bias.detach().cpu().numpy()[ko]
            new_l    = nn.Linear(len(ki), len(ko))
            with torch.no_grad():
                new_l.weight.copy_(torch.tensor(w, dtype=torch.float32))
                new_l.bias.copy_(torch.tensor(b, dtype=torch.float32))
            new_mods.append(new_l.to(DEVICE))
        else:
            new_mods.append(module)
    return nn.Sequential(*new_mods)


# ---------------------------------------------------------------------------
# 7. MÉTRICAS
# ---------------------------------------------------------------------------
def eval_accuracy(model, loader, label=""):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            n       += x.size(0)
    model.train()
    acc = correct / n
    if label:
        print(f"  {label}: {acc:.4%}")
    return acc

def count_flops(model):
    return sum(2 * l.in_features * l.out_features
               for l in model if isinstance(l, nn.Linear))


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Dispositivo: {DEVICE}")
    train_loader = get_loader(train=True)
    test_loader  = get_loader(train=False)

    print("\n" + "=" * 70)
    print(f"ENTRENANDO con GL_LAMBDA={GL_LAMBDA:.0e} + Omega-S")
    print("=" * 70)
    model = build_model()
    model = train(model, train_loader)
    acc_train_full = eval_accuracy(model, train_loader, "acc_train (completo)")
    acc_test_full  = eval_accuracy(model, test_loader,  "acc_test  (completo)")
    flops_full     = count_flops(model)
    print(f"  FLOPs totales: {flops_full:,}")

    print("\n" + "=" * 70)
    print("CALCULANDO ÍNDICES DE PODA (propagación bidireccional)")
    print("=" * 70)
    keep_info = compute_keep_indices(model)
    for info in keep_info:
        orig = info["shape_orig"]
        ni, no = len(info["keep_in"]), len(info["keep_out"])
        print(f"  Capa {info['seq_idx']}: {orig} -> ({no}, {ni})  "
              f"[eliminadas: {orig[1]-ni} cols ({(orig[1]-ni)/orig[1]:.1%}), "
              f"{orig[0]-no} filas ({(orig[0]-no)/orig[0]:.1%})]")

    print("\n" + "=" * 70)
    print("EVALUANDO MODELO PODADO (sin fine-tuning)")
    print("=" * 70)
    pruned = build_pruned_model(model, keep_info)
    acc_train_pruned = eval_accuracy(pruned, train_loader, "acc_train (podado)")
    acc_test_pruned  = eval_accuracy(pruned, test_loader,  "acc_test  (podado)")
    flops_pruned     = count_flops(pruned)

    flop_red      = 1.0 - (flops_pruned / flops_full)
    cost_train    = acc_train_full - acc_train_pruned
    cost_test     = acc_test_full  - acc_test_pruned

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  FLOPs originales:        {flops_full:>12,}")
    print(f"  FLOPs podados:           {flops_pruned:>12,}")
    print(f"  Reducción real de FLOPs: {flop_red:>11.2%}")
    print(f"  Δ accuracy train:        {-cost_train:>+10.4%}")
    print(f"  Δ accuracy test:         {-cost_test:>+10.4%}")

    print("\n" + "=" * 70)
    if cost_test < 0.02 and flop_red > 0.10:
        print("CONCLUSIÓN: PODA VALIDADA DE PUNTA A PUNTA.")
        print(f"  Omega-S + group-lasso (λ={GL_LAMBDA:.0e}) permite eliminar físicamente")
        print(f"  {flop_red:.1%} de los FLOPs con un coste de {cost_test*100:.2f}pp en test.")
        print("  -> Argumento sólido para escalar a GPT-2 small / Llama-3-8B.")
    elif cost_test < 0.02:
        print("CONCLUSIÓN: poda limpia, pero reducción de FLOPs modesta.")
        print("  Sube GL_LAMBDA o añade épocas antes de escalar.")
    else:
        print(f"CONCLUSIÓN: la poda degrada el accuracy {cost_test*100:.2f}pp en test.")
        print("  Opciones:")
        print("    (a) Fine-tuning post-poda: 10-20 steps con LR=1e-4 suele recuperarlo.")
        print("    (b) Usa el punto conservador λ=3e-3 (menos FLOPs pero sin coste).")
    print("=" * 70)
