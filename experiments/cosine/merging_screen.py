#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merging_screen.py  --  CRIBADO ESTATICO de la construccion de interferencia.

Decide si la linea de merging es una linea o una corazonada, ANTES de gastar
celdas de pod. Contesta tres cosas:

  1. La matriz de INTERFERENCIA entre los deltas de dos adaptadores, tiene
     exceso topologico? (C/D despegado de 1). Se prueba con las TRES
     construcciones: lib (sigmoid, la de StochasticOmegaS), raw (sin mapa) y
     coseno.
  2. La SIMETRIZACION lo destruye? An @ Bn^T NO es simetrica (su traspuesta es
     Bn @ An^T, otra matriz), y sobre una matriz asimetrica Tr(A^3) pierde la
     interpretacion de triangulos que motiva el indice. Se comparan tres
     simetrizaciones.
  3. Dice Tr(A^3) algo que los ANGULOS PRINCIPALES no digan ya? Es el control
     que importa: si lo que quieres medir es cuanto interfieren dos subespacios,
     los angulos principales son EXACTOS y directos, y Omega seria un proxy mas
     difuso de algo ya resuelto mejor. Es el mismo criterio con el que se
     descartaron los sectores de redes ingenieriadas.

USO (en el pod, con una GPU libre):
    CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/hf python merging_screen.py

Entrena dos adaptadores LoRA cortos (uno en codigo, otro en prosa), que es lo
que hace falta y no existe: ninguna corrida de rerun_retention.py guarda los
adaptadores, se descartan al terminar cada celda.

Coste: ~10-15 min en un A40.
"""
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/omega-s")

# reutiliza la fontaneria ya verificada de rerun_retention
sys.path.insert(0, "/workspace/omega-s/experiments")
from rerun_retention import (          # noqa: E402
    load_model, get_loader, iter_effective_weights,
    LORA_TARGETS, LORA_R, LORA_ALPHA, DEVICE, GRAD_ACCUM, LR,
)

STEPS = int(os.environ.get("MERGE_STEPS", "300"))
# DOS SEMILLAS DISTINTAS, y no es un detalle: con la misma semilla lora_A se
# inicializa IGUAL en los dos adaptadores, y como delta = B@A su espacio de
# filas queda contenido en el de A, el mismo para ambos. Los angulos
# principales salen 0 por construccion y el test no mide nada.
SEED_A = int(os.environ.get("MERGE_SEED_A", "42"))
SEED_B = int(os.environ.get("MERGE_SEED_B", "4242"))
EPS = 1e-6
OUT = os.environ.get("OUT", "merging_screen.json")


# ---------------------------------------------------------------------------
# 1. entrenar dos adaptadores cortos y quedarse SOLO con los deltas
# ---------------------------------------------------------------------------
def train_delta(domain, seed, steps=STEPS):
    """Entrena LoRA en un dominio y devuelve {nombre_modulo: delta B@A}."""
    print(f"  entrenando adaptador en '{domain}' semilla {seed} ({steps} pasos)...",
          flush=True)
    model, tok = load_model(seed, smoke=False)
    loader = get_loader(tok, domain, smoke=False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    step = 0
    for b in loader:
        if step >= steps:
            break
        loss = model(input_ids=b["input_ids"].to(DEVICE),
                     labels=b["labels"].to(DEVICE)).loss / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        step += 1

    scaling = LORA_ALPHA / LORA_R
    deltas = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not any(name.endswith(m) for m in LORA_TARGETS):
                continue
            if not hasattr(mod, "lora_A"):
                continue
            A = mod.lora_A["default"].weight.detach().float()
            B = mod.lora_B["default"].weight.detach().float()
            deltas[name] = (scaling * (B @ A)).cpu()
    del model
    torch.cuda.empty_cache()
    print(f"    {len(deltas)} modulos", flush=True)
    return deltas


# ---------------------------------------------------------------------------
# 2. las tres construcciones sobre la matriz de interferencia
# ---------------------------------------------------------------------------
def interference_raw(dA, dB):
    """Producto cruzado crudo: dA @ dB^T. Sin mapa, sin normalizar."""
    return (dA @ dB.t()).abs()


def interference_lib(dA, dB):
    """Como haria StochasticOmegaS: sigmoid sobre el valor absoluto."""
    return torch.sigmoid((dA @ dB.t()).abs())


def interference_cos(dA, dB, eps=1e-8):
    """Coseno CRUZADO entre las filas de los dos deltas. Es el gancho."""
    An = dA / (dA.norm(dim=1, keepdim=True) + eps)
    Bn = dB / (dB.norm(dim=1, keepdim=True) + eps)
    return (An @ Bn.t()).abs().clamp(0.0, 1.0)


def symmetrize(A, how):
    if how == "media":
        return (A + A.t()) / 2.0
    if how == "max":
        return torch.maximum(A, A.t())
    if how == "AAt":            # bipartito: A A^T, simetrica por construccion
        return A @ A.t()
    raise ValueError(how)


def excess_CD(A, zero_diag=True):
    """C/D exacto, el mismo estadistico de toda la ronda. 1.0 = sin exceso."""
    A = A.double().clone()
    if zero_diag:
        A.fill_diagonal_(0.0)
    n = A.shape[0]
    A2 = A @ A
    denom = A2.sum() - torch.diagonal(A2).sum()
    C = (A * A2).sum() / (denom + 1e-12)
    off = A.sum() - torch.diagonal(A).sum()
    D = off / (n * (n - 1) + 1e-12)
    return float(C / (D + 1e-12))


# ---------------------------------------------------------------------------
# 3. el control clasico: angulos principales entre los dos subespacios
# ---------------------------------------------------------------------------
def principal_angles(dA, dB, r=None):
    """Angulos principales entre los espacios de FILAS de rango r de los deltas.

    OJO, ERROR CORREGIDO: usar torch.linalg.qr sobre dA.t() devuelve una base
    COMPLETA cuando la matriz es cuadrada (q_proj es 4096x4096), no el
    subespacio de rango r. El producto de dos bases completas es ortogonal y
    todos sus valores singulares valen 1, o sea angulos 0 POR CONSTRUCCION.
    El espacio de filas real de B@A tiene dimension r (8), asi que hay que
    sacarlo con SVD y quedarse con los r primeros vectores singulares derechos.
    """
    r = r or LORA_R
    Va = torch.linalg.svd(dA.double(), full_matrices=False).Vh[:r]   # (r, in)
    Vb = torch.linalg.svd(dB.double(), full_matrices=False).Vh[:r]
    s = torch.linalg.svdvals(Va @ Vb.t()).clamp(0, 1)
    ang = torch.rad2deg(torch.arccos(s))
    return s.tolist(), ang.tolist()


# ---------------------------------------------------------------------------
def main():
    print("CRIBADO DE MERGING: interferencia entre dos adaptadores\n")
    print(f"modulos LoRA: {LORA_TARGETS}   rango {LORA_R}   pasos {STEPS}")
    print(f"semillas: A={SEED_A}  B={SEED_B}  (DISTINTAS a proposito)\n")

    cache = os.environ.get("DELTA_CACHE", "merging_deltas.pt")
    if os.path.exists(cache) and os.environ.get("USE_CACHE", "1") == "1":
        print(f"  reutilizando deltas de {cache} (no se reentrena)")
        blob = torch.load(cache)
        dA, dB = blob["A"], blob["B"]
    else:
        dA = train_delta("code", SEED_A)
        dB = train_delta("prose", SEED_B)
        torch.save({"A": dA, "B": dB}, cache)
        print(f"  deltas guardados en {cache}")

    comunes = sorted(set(dA) & set(dB))
    print(f"\nmodulos comunes: {len(comunes)}")
    # cuatro modulos de capas iniciales, que es donde vive la senal
    muestra = comunes[:4]

    builds = {"raw": interference_raw, "lib": interference_lib,
              "cos": interference_cos}
    filas = []
    print("\n" + "=" * 78)
    print("EXCESO TOPOLOGICO C/D DE LA MATRIZ DE INTERFERENCIA  (1.0 = sin exceso)")
    print("=" * 78)
    print(f"{'modulo':38} {'constr':6} {'media':>9} {'max':>9} {'AAt':>9}")
    for nm in muestra:
        a, b = dA[nm], dB[nm]
        for cname, fn in builds.items():
            M = fn(a, b)
            vals = {}
            for how in ("media", "max", "AAt"):
                try:
                    vals[how] = excess_CD(symmetrize(M, how))
                except Exception as e:
                    vals[how] = float("nan")
            corto = nm.replace("base_model.model.model.layers.", "L")
            print(f"{corto:38} {cname:6} {vals['media']:9.4f} "
                  f"{vals['max']:9.4f} {vals['AAt']:9.4f}")
            filas.append(dict(modulo=nm, constr=cname, **vals))
        print()

    print("=" * 78)
    print("CONTROL CLASICO: angulos principales entre los espacios de filas")
    print("=" * 78)
    ctrl = []
    for nm in muestra:
        s, ang = principal_angles(dA[nm], dB[nm])
        corto = nm.replace("base_model.model.model.layers.", "L")
        print(f"{corto:38} cos: " + " ".join(f"{v:.3f}" for v in s[:6]))
        print(f"{'':38} deg: " + " ".join(f"{v:5.1f}" for v in ang[:6]))
        ctrl.append(dict(modulo=nm, cos=s, deg=ang))

    json.dump(dict(excess=filas, angles=ctrl), open(OUT, "w"), indent=2)

    print("\n" + "=" * 78)
    print("COMO LEERLO")
    print("=" * 78)
    print("1. Si NINGUNA construccion despega C/D de 1.0, la interferencia no")
    print("   tiene estructura triadica y la linea de merging se CIERRA aqui,")
    print("   por cero euros de pod.")
    print("2. Si solo despega con una simetrizacion concreta, hay que declarar")
    print("   esa eleccion como parte de la construccion, no como detalle.")
    print("3. Mira los angulos principales: si ya separan bien los modulos que")
    print("   interfieren de los que no, Omega seria un proxy mas difuso de algo")
    print("   que la SVD resuelve exacto, y la linea no tiene hueco aunque el")
    print("   exceso sea distinto de 1.")
    print(f"\nGuardado en {OUT}")


if __name__ == "__main__":
    main()
