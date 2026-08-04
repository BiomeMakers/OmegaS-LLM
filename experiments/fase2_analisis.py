# =============================================================================
# Omega-S : Reproducibility Script
# Author:  Alberto Acedo (acedo@biomemakers.com)
# License: AGPL-3.0 (research use). See LICENSE in the repository root.
# Patent:  USPTO Patent Pending
# Repo:    https://github.com/BiomeMakers/OmegaS-LLM
# =============================================================================

"""
FASE 2 : Análisis final
========================
Ejecutar DESPUÉS de que ambas GPUs hayan terminado.
Lee results_baseline.json y results_omega.json y genera la tabla del paper.

Uso: python fase2_analisis.py
"""

import json
import pandas as pd

def load(path):
    with open(path) as f:
        return json.load(f)

def print_table(res_b, res_o):
    metrics = [
        ("HumanEval baseline",        "he_baseline"),
        ("HumanEval post-código",      "he_post_code"),
        ("HumanEval post-prosa",       "he_post_prose"),
        ("Olvido (↓ HumanEval)",       "forgetting"),
        ("Retención (%)",              "retention_pct"),
        ("PPL código post-prosa (↓)",  "ppl_code_post_prose"),
        ("PPL prosa post-prosa (↓)",   "ppl_prose_post_prose"),
    ]

    print("\n" + "=" * 85)
    print(" TABLA FINAL : OLVIDO CATASTRÓFICO EN LLAMA-3-8B")
    print("=" * 85)
    print(f"{'Métrica':<35} {'Baseline LoRA':>15} {'Omega-S LoRA':>15} "
          f"{'Δ':>15}")
    print("-" * 85)

    rows = []
    for label, key in metrics:
        vb = res_b.get(key, 0)
        vo = res_o.get(key, 0)
        d  = vo - vb
        pct_keys = {"he_baseline", "he_post_code", "he_post_prose",
                    "forgetting", "retention_pct"}
        fmt = ".2%" if key in pct_keys else ".2f"
        print(f"{label:<35} {vb:>15{fmt}} {vo:>15{fmt}} {d:>+15{fmt}}")
        rows.append({"métrica": label, "baseline": vb,
                     "omega_s": vo, "delta": d})

    print("=" * 85)

    ri = res_o["retention_pct"] - res_b["retention_pct"]
    fr = res_b["forgetting"] - res_o["forgetting"]

    print("\nVEREDICTO")
    print("=" * 85)
    if ri > 0.10:
        print(f"✓ Omega-S mejora retención en {ri:.1%} : resultado FUERTE para el paper.")
        print(f"✓ Reduce olvido catastrófico en {fr:.1%} pp absolutos.")
        print("→ Actualizar preprint Sección 4.4 con estos números.")
        print("→ El paper está listo para NeurIPS workshop / TMLR submission.")
    elif ri > 0.05:
        print(f"✓ Omega-S mejora retención en {ri:.1%} : resultado SÓLIDO.")
        print(f"✓ Reduce olvido en {fr:.1%} pp.")
        print("→ Resultado publicable. Considerar más épocas para ampliar señal.")
    elif ri > 0:
        print(f"~ Mejora marginal ({ri:.1%}). Señal positiva pero débil.")
        print("→ Subir MAX_SAMPLES a 10000 o EPOCHS a 2 antes de publicar.")
    else:
        print(f"✗ No hay mejora de retención en esta configuración.")
        print("→ Revisar lambda Omega-S y frecuencia K.")

    print("=" * 85)

    df = pd.DataFrame(rows)
    df.to_csv("fase2_tabla_paper.csv", index=False)
    print("\nTabla guardada en fase2_tabla_paper.csv")
    return df

if __name__ == "__main__":
    try:
        res_b = load("results_baseline.json")
        res_o = load("results_omega.json")
        print_table(res_b, res_o)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Asegúrate de que ambas GPUs han terminado antes de correr el análisis.")
