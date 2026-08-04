# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
TEST 6 (v3): Control : Weight Decay vs. Omega-S + Group-Lasso
=============================================================================
Corrección v3: propagación de poda en tres pasadas para cubrir todos los
casos (poda por filas, por columnas, o mixta):
  1. Hacia adelante: filas muertas de capa i -> columnas a eliminar en capa i+1
  2. Hacia atrás:   columnas eliminadas en capa i+1 -> filas a eliminar en capa i
  3. Segunda pasada adelante para reconciliar cualquier inconsistencia residual
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
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SEED           = 42
EPOCHS         = 5
BATCH_SIZE     = 128
LR             = 1e-3
WD_GRID        = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
OMEGA_LAMBDA   = 0.05
OMEGA_EVERY_K  = 10
OMEGA_N_PROBES = 4
GL_LAMBDA      = 1e-2
ZERO_THRESHOLD = 1e-3
DEAD_FRACTION  = 0.95

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
def train_wd(model, loader, wd_lambda):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        correct, n = 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            l2     = sum(p.pow(2).sum() for p in model.parameters())
            loss   = loss + wd_lambda * l2
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            n       += x.size(0)
    return model

def train_omega_gl(model, loader):
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
    return model


# ---------------------------------------------------------------------------
# 5. PODA CON PROPAGACIÓN EN TRES PASADAS
# ---------------------------------------------------------------------------
def dead_col_idx(w):
    return np.where((np.abs(w) < ZERO_THRESHOLD).mean(axis=0) >= DEAD_FRACTION)[0]

def dead_row_idx(w):
    return np.where((np.abs(w) < ZERO_THRESHOLD).mean(axis=1) >= DEAD_FRACTION)[0]

def compute_keep_indices(model):
    """
    Tres pasadas para garantizar consistencia en cualquier patrón de sparsity:
    Pasada 1 (→): filas muertas de capa i se convierten en cols a eliminar en i+1
    Pasada 2 (←): cols eliminadas en capa i+1 obligan a eliminar esas filas en capa i
    Pasada 3 (→): segunda reconciliación adelante para cerrar cualquier residuo
    La capa de salida nunca poda filas (no eliminar clases).
    """
    ll      = get_linear_layers(model)
    n       = len(ll)
    weights = [l.weight.detach().cpu().numpy() for _, l in ll]

    # Inicializar con muertes locales
    eliminate_cols = [set(dead_col_idx(w).tolist()) for w in weights]
    eliminate_rows = [set(dead_row_idx(w).tolist()) for w in weights]
    eliminate_rows[-1] = set()  # no podar clases de salida
    eliminate_cols[0]  = set()  # no podar entradas de capa 0: son píxeles del input,
                                 # dimensión fija del dataset : no se pueden eliminar
                                 # sin cambiar el preprocesado de los datos

    # Pasada 1 → : filas muertas de capa i => cols a eliminar en capa i+1
    for i in range(n - 1):
        eliminate_cols[i + 1].update(eliminate_rows[i])

    # Pasada 2 ← : cols eliminadas en capa i+1 => filas a eliminar en capa i
    for i in range(n - 2, -1, -1):
        eliminate_rows[i].update(eliminate_cols[i + 1])

    # Pasada 3 → : propagar de nuevo por si pasada 2 añadió filas nuevas
    for i in range(n - 1):
        eliminate_cols[i + 1].update(eliminate_rows[i])

    keep_in  = [np.setdiff1d(np.arange(weights[i].shape[1]),
                              np.array(sorted(eliminate_cols[i])))
                for i in range(n)]
    keep_out = [np.setdiff1d(np.arange(weights[i].shape[0]),
                              np.array(sorted(eliminate_rows[i])))
                for i in range(n)]

    # Verificación de consistencia antes de construir el modelo
    for i in range(n - 1):
        assert len(keep_out[i]) == len(keep_in[i + 1]), (
            f"Inconsistencia entre capa {ll[i][0]} (salida {len(keep_out[i])}) "
            f"y capa {ll[i+1][0]} (entrada {len(keep_in[i+1])})"
        )

    return [{"seq_idx":    ll[i][0],
             "keep_in":    keep_in[i],
             "keep_out":   keep_out[i],
             "shape_orig": weights[i].shape}
            for i in range(n)]

def build_pruned_model(model, keep_info):
    layer_map = {info["seq_idx"]: info for info in keep_info}
    new_mods  = []
    for seq_idx, module in enumerate(model):
        if isinstance(module, nn.Linear):
            info   = layer_map[seq_idx]
            ki, ko = info["keep_in"], info["keep_out"]
            w = module.weight.detach().cpu().numpy()[np.ix_(ko, ki)]
            b = module.bias.detach().cpu().numpy()[ko]
            new_l = nn.Linear(len(ki), len(ko))
            with torch.no_grad():
                new_l.weight.copy_(torch.tensor(w, dtype=torch.float32))
                new_l.bias.copy_(torch.tensor(b, dtype=torch.float32))
            new_mods.append(new_l.to(DEVICE))
        else:
            new_mods.append(module)
    return nn.Sequential(*new_mods)

def run_pruning_pipeline(model, label=""):
    keep_info    = compute_keep_indices(model)
    pruned       = build_pruned_model(model, keep_info)
    flops_full   = sum(2*l.in_features*l.out_features
                       for l in model  if isinstance(l, nn.Linear))
    flops_pruned = sum(2*l.in_features*l.out_features
                       for l in pruned if isinstance(l, nn.Linear))
    if label:
        print(f"  Estructura podada ({label}):")
    for info in keep_info:
        o = info["shape_orig"]
        ni, no = len(info["keep_in"]), len(info["keep_out"])
        print(f"    Capa {info['seq_idx']}: {o} -> ({no},{ni})  "
              f"[cols -{o[1]-ni} ({(o[1]-ni)/o[1]:.1%}) | "
              f"filas -{o[0]-no} ({(o[0]-no)/o[0]:.1%})]")
    return pruned, flops_full, flops_pruned

def eval_acc(model, loader):
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
# 6. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Dispositivo: {DEVICE}")
    train_loader = get_loader(train=True)
    test_loader  = get_loader(train=False)

    # --- REFERENCIA: Omega-S + GL ---
    print("\n" + "=" * 70)
    print(f"REFERENCIA: Omega-S + Group-Lasso (GL λ={GL_LAMBDA:.0e})")
    print("=" * 70)
    torch.manual_seed(SEED)
    model_omega = build_model()
    model_omega = train_omega_gl(model_omega, train_loader)
    acc_omega   = eval_acc(model_omega, test_loader)
    print(f"  acc_test (completo): {acc_omega:.4%}")
    pruned_omega, ff_o, fp_o = run_pruning_pipeline(model_omega, "Omega+GL")
    acc_omega_pruned = eval_acc(pruned_omega, test_loader)
    flop_red_omega   = 1.0 - fp_o / ff_o
    print(f"  acc_test (podado):   {acc_omega_pruned:.4%}  |  "
          f"reducción FLOPs: {flop_red_omega:.2%}")

    # --- BARRIDO WD ---
    print("\n" + "=" * 70)
    print(f"BARRIDO WD (Adam + L2) buscando acc_test ≈ {acc_omega:.2%}")
    print("=" * 70)
    wd_results = []
    for wd_lambda in WD_GRID:
        torch.manual_seed(SEED)
        m   = build_model()
        m   = train_wd(m, train_loader, wd_lambda)
        acc = eval_acc(m, test_loader)
        sp  = sum((np.abs(l.weight.detach().cpu().numpy()) < ZERO_THRESHOLD).mean()
                  for _, l in get_linear_layers(m)) / len(get_linear_layers(m))
        print(f"  WD λ={wd_lambda:.0e}  ->  acc_test={acc:.4%}  |  sparsity≈{sp:.2%}")
        wd_results.append({"lambda": wd_lambda, "acc": acc, "model": m})

    best_wd = min(wd_results, key=lambda r: abs(r["acc"] - acc_omega))
    print(f"\n  → λ seleccionado: {best_wd['lambda']:.0e}  (acc={best_wd['acc']:.4%})")

    # --- PODA WD ---
    print("\n" + "=" * 70)
    print(f"PODA sobre WD λ={best_wd['lambda']:.0e}")
    print("=" * 70)
    pruned_wd, ff_wd, fp_wd = run_pruning_pipeline(best_wd["model"], "WD")
    acc_wd_pruned = eval_acc(pruned_wd, test_loader)
    flop_red_wd   = 1.0 - fp_wd / ff_wd
    print(f"  acc_test (podado): {acc_wd_pruned:.4%}  |  "
          f"reducción FLOPs: {flop_red_wd:.2%}")

    # --- TABLA FINAL ---
    print("\n\n" + "=" * 95)
    print("TABLA COMPARATIVA FINAL")
    print("=" * 95)
    print(f"{'Método':<30} {'Acc completo':>13} {'Acc podado':>12} "
          f"{'Δ acc':>8} {'FLOPs orig':>11} {'FLOPs podado':>13} {'Reducción':>10}")
    print("-" * 95)
    for nombre, acc_f, acc_p, ff, fp, fr in [
        ("Omega-S + GL",       acc_omega,        acc_omega_pruned, ff_o,  fp_o,  flop_red_omega),
        (f"WD λ={best_wd['lambda']:.0e}", best_wd["acc"], acc_wd_pruned,   ff_wd, fp_wd, flop_red_wd),
    ]:
        print(f"{nombre:<30} {acc_f:>13.4%} {acc_p:>12.4%} "
              f"{acc_p-acc_f:>+8.4%} {ff:>11,} {fp:>13,} {fr:>10.2%}")
    print("=" * 95)

    # --- CONCLUSIÓN ---
    diff_flop = flop_red_omega - flop_red_wd
    diff_acc  = abs(acc_omega - best_wd["acc"])
    print("\nCONCLUSIÓN")
    print("=" * 95)
    if diff_acc > 0.02:
        print(f"[Nota] El barrido no encontró un λ de WD dentro de ±2pp del accuracy "
              f"de referencia ({acc_omega:.2%}). El más cercano fue "
              f"{best_wd['acc']:.2%} (λ={best_wd['lambda']:.0e}).")
        print("Esto en sí es un dato: WD necesita un λ muy agresivo para bajar al")
        print("nivel de accuracy de Omega-S+GL, lo que refleja la diferencia en cómo")
        print("cada regularizador presiona la red.\n")
    if diff_flop > 0.10:
        print(f"Con accuracy de partida comparable, Omega-S+GL habilita {diff_flop:.1%} más")
        print(f"de reducción de FLOPs que WD ({flop_red_omega:.2%} vs {flop_red_wd:.2%}).")
        print("-> COMPARACIÓN DEFINITIVA: mismo pipeline, regularizador distinto,")
        print("   resultado radicalmente distinto. Argumento blindado para el CTO.")
    elif diff_flop > 0.03:
        print(f"Omega-S+GL produce {diff_flop:.1%} más de reducción de FLOPs que WD.")
        print("Ventaja real pero moderada a esta escala. Escalar a GPT-2 small.")
    else:
        print(f"Ambos producen reducciones similares ({flop_red_omega:.1%} vs {flop_red_wd:.1%}).")
        print("Revisar narrativa: la ventaja de Omega-S puede estar en accuracy, no en poda.")
    print("=" * 95)
