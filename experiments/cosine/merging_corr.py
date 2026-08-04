#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merging_corr.py  --  ¿ve Tr(A^3) algo que el primer angulo principal no vea?

Reutiliza los deltas cacheados por merging_screen.py (merging_deltas.pt), asi
que NO reentrena nada: son segundos.

Para los 64 modulos (q_proj y v_proj en las 32 capas) calcula:
  - primer coseno principal entre los espacios de filas de rango r de los dos
    deltas. Es la medida EXACTA de solape de subespacio, una SVD
  - exceso topologico C/D de la matriz de interferencia cruda, con las
    simetrizaciones media y bipartita A·A^T (las dos que dieron mas senal)

Y correlaciona. LA PREGUNTA: si el exceso triadico y el primer angulo son la
misma informacion, Omega es redundante en merging y la linea se cierra con
fundamento. Si no correlacionan, Omega ve otra cosa, y entonces la pregunta
siguiente es si esa otra cosa predice la calidad de la fusion.

CONTEXTO MEDIDO: la interferencia entre dos adaptadores es de RANGO 1. El
primer angulo esta 3-5x por encima del azar (z de +20 a +37) y del segundo en
adelante son indistinguibles del azar (referencia simulada: primer coseno
medio 0.0770 entre subespacios de dim 8 en R^4096).

USO:
    CUDA_VISIBLE_DEVICES=0 python merging_corr.py
"""
import os
import sys
import json

import torch

sys.path.insert(0, "/workspace/omega-s")
sys.path.insert(0, "/workspace/omega-s/experiments")
from rerun_retention import LORA_R, DEVICE      # noqa: E402

CACHE = os.environ.get("DELTA_CACHE", "merging_deltas.pt")
OUT = os.environ.get("OUT", "merging_corr.json")
# referencia de azar para dim 8 en R^4096, simulada con 300 pares
AZAR_COS1 = 0.0770


def top_cos(dA, dB, r=LORA_R):
    """Primer coseno principal entre los espacios de filas de rango r."""
    Va = torch.linalg.svd(dA, full_matrices=False).Vh[:r]
    Vb = torch.linalg.svd(dB, full_matrices=False).Vh[:r]
    return float(torch.linalg.svdvals(Va @ Vb.t()).clamp(0, 1)[0])


def excess_CD(A):
    """C/D exacto, diagonal a cero. 1.0 = sin exceso triadico."""
    A = A.clone()
    A.fill_diagonal_(0.0)
    n = A.shape[0]
    A2 = A @ A
    C = (A * A2).sum() / (A2.sum() - torch.diagonal(A2).sum() + 1e-12)
    D = (A.sum() - torch.diagonal(A).sum()) / (n * (n - 1) + 1e-12)
    return float(C / (D + 1e-12))


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos + 1.0
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db + 1e-12)


def spearman(a, b):
    return pearson(rank(a), rank(b))


def main():
    if not os.path.exists(CACHE):
        sys.exit(f"No encuentro {CACHE}. Corre antes merging_screen.py.")
    blob = torch.load(CACHE)
    dA, dB = blob["A"], blob["B"]
    comunes = sorted(set(dA) & set(dB))
    print(f"modulos comunes: {len(comunes)}   rango LoRA: {LORA_R}")
    print(f"referencia de azar del primer coseno: {AZAR_COS1:.4f}\n")

    filas = []
    print(f"{'modulo':34} {'cos1':>7} {'xazar':>6} {'C/D media':>10} {'C/D AAt':>9}")
    for nm in comunes:
        a = dA[nm].to(DEVICE, torch.float64)
        b = dB[nm].to(DEVICE, torch.float64)
        c1 = top_cos(a, b)
        M = (a @ b.t()).abs()
        e_med = excess_CD((M + M.t()) / 2.0)
        e_aat = excess_CD(M @ M.t())
        corto = nm.replace("base_model.model.model.layers.", "L")
        print(f"{corto:34} {c1:7.4f} {c1/AZAR_COS1:6.1f} {e_med:10.4f} {e_aat:9.4f}")
        filas.append(dict(modulo=nm, cos1=c1, cd_media=e_med, cd_aat=e_aat))
        del a, b, M
        torch.cuda.empty_cache()

    c1s = [f["cos1"] for f in filas]
    med = [f["cd_media"] for f in filas]
    aat = [f["cd_aat"] for f in filas]

    print("\n" + "=" * 72)
    print("CORRELACION ENTRE EL EXCESO TRIADICO Y EL PRIMER ANGULO PRINCIPAL")
    print("=" * 72)
    for lbl, v in (("C/D media", med), ("C/D AAt", aat)):
        print(f"  {lbl:10} vs cos1   Pearson {pearson(c1s, v):+.3f}   "
              f"Spearman {spearman(c1s, v):+.3f}")

    # control: correlacionan las dos simetrizaciones entre si?
    print(f"\n  (control) C/D media vs C/D AAt   Spearman {spearman(med, aat):+.3f}")

    # cuantos modulos superan el azar
    n_sig = sum(1 for c in c1s if c > 2 * AZAR_COS1)
    print(f"\n  modulos con solape claramente sobre el azar (cos1 > 2x): "
          f"{n_sig} de {len(c1s)}")

    json.dump(filas, open(OUT, "w"), indent=2)
    print("\n" + "=" * 72)
    print("COMO LEERLO")
    print("=" * 72)
    print("  |Spearman| alto (>0.7): el exceso triadico y el angulo principal son")
    print("    la MISMA informacion. Omega es redundante en merging y la linea se")
    print("    cierra con fundamento, no por corazonada.")
    print("  |Spearman| bajo (<0.3): Omega ve algo DISTINTO del solape de")
    print("    subespacio. No lo valida, pero abre la pregunta siguiente: predice")
    print("    ese algo la calidad de la fusion? Eso ya seria otra ronda con coste.")
    print("  Intermedio: informacion parcialmente solapada, y entonces el juicio")
    print("    es de coste-beneficio, no de principio.")
    print(f"\nGuardado en {OUT}")


if __name__ == "__main__":
    main()
