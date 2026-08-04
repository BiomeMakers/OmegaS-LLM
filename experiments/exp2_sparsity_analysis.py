# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
TEST 1 (Parte 1 del hilo): ¿La sparsity del 5.1% de Omega-S es estructurada?
=============================================================================
Objetivo: responder la pregunta previa que determina si vale la pena perseguir
"Omega-S -> menos energía en inferencia". Si los ceros caen en filas/canales
completos, son podables de verdad (ahorro real de FLOPs). Si están dispersos
al azar, la sparsity es casi inútil en GPU estándar sin kernels especiales.

Cómo usarlo en Colab:
1. Sube tu checkpoint de "Red B" (el modelo entrenado con Omega-S) a Colab.
2. Cambia CHECKPOINT_PATH y, si tu arquitectura no es la de ejemplo (MLP),
   sustituye `build_model()` por la construcción real de tu red.
3. Ejecuta. El script no necesita GPU, corre bien en CPU de Colab.

No requiere librerías fuera de lo estándar (torch, numpy).
"""

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# 0. CONFIGURACIÓN : edita esto para tu caso real
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "red_b_omega_s.pt"   # ruta a tu checkpoint entrenado
ZERO_THRESHOLD = 1e-3                  # por debajo de esto se considera "cero"
STRUCTURED_DEAD_FRACTION = 0.95        # % de la fila/columna que debe ser ~0
                                        # para contarla como canal muerto


def build_model():
    """
    Reemplaza esto por la arquitectura real de tu Red B (Split-MNIST u otra).
    Se deja un MLP de ejemplo para que el script corra de punta a punta
    aunque aún no hayas conectado tu checkpoint real.
    """
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def load_weights(model, path):
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Checkpoint cargado desde {path}")
    except FileNotFoundError:
        print(f"[AVISO] No se encontró {path}. Usando pesos aleatorios "
              f"solo para validar que el script corre. Sustituye por tu "
              f"checkpoint real antes de sacar conclusiones.")
    return model


# ---------------------------------------------------------------------------
# 1. ANÁLISIS DE ESTRUCTURA DE LA SPARSITY
# ---------------------------------------------------------------------------
def analyze_layer_sparsity(weight: torch.Tensor, name: str):
    """
    weight: tensor 2D (out_features, in_features) de una capa Linear.
    Compara sparsity total vs. sparsity "estructurada" (filas/columnas
    enteras casi a cero -> canales podables).
    """
    w = weight.detach().cpu().numpy()
    total_elems = w.size
    zero_mask = np.abs(w) < ZERO_THRESHOLD
    total_sparsity = zero_mask.mean()

    # Fracción de ceros por fila (neurona de salida) y por columna (entrada)
    row_zero_frac = zero_mask.mean(axis=1)   # por neurona de salida
    col_zero_frac = zero_mask.mean(axis=0)   # por neurona de entrada

    dead_rows = (row_zero_frac >= STRUCTURED_DEAD_FRACTION).sum()
    dead_cols = (col_zero_frac >= STRUCTURED_DEAD_FRACTION).sum()

    print(f"\n--- Capa: {name} ---")
    print(f"  Forma: {w.shape}")
    print(f"  Sparsity total (elemento a elemento): {total_sparsity:.4%}")
    print(f"  Filas de salida ~muertas ({STRUCTURED_DEAD_FRACTION:.0%} ceros o más): "
          f"{dead_rows}/{w.shape[0]} ({dead_rows / w.shape[0]:.2%})")
    print(f"  Columnas de entrada ~muertas: "
          f"{dead_cols}/{w.shape[1]} ({dead_cols / w.shape[1]:.2%})")

    is_structured = (dead_rows / w.shape[0] > 0.02) or (dead_cols / w.shape[1] > 0.02)
    veredicto = "ESTRUCTURADA (podable)" if is_structured else "DISPERSA (poco aprovechable)"
    print(f"  Veredicto preliminar: {veredicto}")

    return {
        "name": name,
        "total_sparsity": total_sparsity,
        "dead_row_frac": dead_rows / w.shape[0],
        "dead_col_frac": dead_cols / w.shape[1],
        "structured": is_structured,
    }


def count_linear_flops(model, input_dim):
    """FLOPs aproximados (2*in*out por capa Linear, ignorando activaciones)."""
    total = 0
    dim = input_dim
    for layer in model:
        if isinstance(layer, nn.Linear):
            total += 2 * layer.in_features * layer.out_features
    return total


def prune_structured(model, results, input_dim):
    """
    Reconstruye una versión más pequeña del modelo eliminando físicamente
    las filas/columnas muertas detectadas. Devuelve el modelo podado y sus
    FLOPs, para comparar contra el original.
    NOTA: implementación simplificada para capas Linear en Sequential;
    si tu arquitectura tiene skip-connections o attention, habrá que
    adaptar la lógica de qué canales se pueden eliminar sin romper shapes.
    """
    linears = [l for l in model if isinstance(l, nn.Linear)]
    new_layers = []
    keep_in = None  # índices de entrada a mantener (de la poda de la capa anterior)

    for i, layer in enumerate(linears):
        w = layer.weight.detach().cpu().numpy()
        zero_mask = np.abs(w) < ZERO_THRESHOLD
        row_zero_frac = zero_mask.mean(axis=1)
        keep_out = np.where(row_zero_frac < STRUCTURED_DEAD_FRACTION)[0]

        if keep_in is None:
            keep_in = np.arange(w.shape[1])

        new_w = w[np.ix_(keep_out, keep_in)]
        new_layer = nn.Linear(len(keep_in), len(keep_out))
        with torch.no_grad():
            new_layer.weight.copy_(torch.tensor(new_w))
            new_layer.bias.copy_(layer.bias[keep_out])
        new_layers.append(new_layer)
        new_layers.append(nn.ReLU())
        keep_in = keep_out  # la salida podada de esta capa es la entrada de la siguiente

    new_layers = new_layers[:-1]  # quita el último ReLU sobrante
    pruned_model = nn.Sequential(*new_layers)
    return pruned_model


# ---------------------------------------------------------------------------
# 2. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_model()
    model = load_weights(model, CHECKPOINT_PATH)

    print("=" * 70)
    print("ANÁLISIS DE ESTRUCTURA DE SPARSITY")
    print("=" * 70)

    results = []
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            r = analyze_layer_sparsity(layer.weight, f"Linear_{i}")
            results.append(r)

    any_structured = any(r["structured"] for r in results)

    print("\n" + "=" * 70)
    if not any_structured:
        print("CONCLUSIÓN: la sparsity parece NO estructurada.")
        print("-> El camino de 'Omega-S baja energía vía pruning' probablemente")
        print("   no da fruto sin cambios adicionales. Considera cerrar esta línea.")
    else:
        print("CONCLUSIÓN: hay sparsity estructurada real. Procediendo a podar...")
        original_flops = count_linear_flops(model, 784)
        pruned_model = prune_structured(model, results, 784)
        pruned_flops = count_linear_flops(pruned_model, 784)
        reduction = 1 - (pruned_flops / original_flops)
        print(f"  FLOPs originales: {original_flops:,}")
        print(f"  FLOPs tras poda:  {pruned_flops:,}")
        print(f"  Reducción de FLOPs: {reduction:.2%}")
        print("\n  Recuerda: valida accuracy del modelo podado antes de reportar")
        print("  esta cifra como 'ahorro de energía sin pérdida de calidad'.")
    print("=" * 70)
