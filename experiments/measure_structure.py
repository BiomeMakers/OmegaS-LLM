#!/usr/bin/env python3
"""
=============================================================================
EXPERIMENTO: ¿la ventaja de Omega-S sobre EWC depende de la estructura de red?
CRITERIO DE FALSACION ESCRITO ANTES DE VER NINGUN DATO (25-jul)
=============================================================================

HIPOTESIS PRINCIPAL (declarada antes):
  El gain de omega_lib sobre EWC por semilla correlaciona POSITIVAMENTE con
  cuanto SUBE C (clustering, Tr(A^3) normalizado) al aplicar Omega, medido
  como C_omega - C_none. Es decir: Omega gana en las semillas donde de verdad
  hace topologia (sube el clustering), no donde solo iguala normas.
  -> Si se cumple: el mecanismo es TOPOLOGICO.

HIPOTESIS RIVAL (declarada antes):
  El gain correlaciona con cuanto BAJA Coex (varianza de grados), medido como
  Coex_none - Coex_omega. Coex correlaciona 0.79 con normas de fila.
  -> Si se cumple: el mecanismo es IGUALAR NORMAS DE FILA.

Las dos hipotesis son mutuamente excluyentes en su INTERPRETACION: C esta en
el numerador de Omega (lo sube), Coex en el denominador (lo baja). Cual de los
dos movimientos explica el gain es lo que decide la naturaleza del mecanismo.

EXPLORATORIAS (no principales, no deciden nada, solo generan hipotesis):
  D (densidad), M (modularidad), y la brecha grado-vs-norma-de-fila.

CORRECCION POR MULTIPLES COMPARACIONES:
  Medimos varias metricas. Umbral de significancia Bonferroni: 0.05 / 5 = 0.01
  para declarar cualquier correlacion como robusta. Con 10 semillas esto es
  exigente A PROPOSITO: nos protege de cantar victoria con ruido.

FALSACION EXPLICITA:
  Si NI C NI Coex correlacionan con el gain por encima del umbral corregido,
  entonces el patron bimodal observado en la tabla de 10 semillas era RUIDO,
  no un mecanismo dependiente de estructura, y la hipotesis del mecanismo doble
  NO se sostiene sobre estos datos. Se reporta como negativo, no se reencuadra.

QUE MIDE:
  Para cada semilla, reentrena SOLO la tarea A (codigo) en dos brazos (none y
  omega_lib), y sobre las matrices LoRA efectivas justo antes de la tarea B
  calcula C, D, M, Coex y la brecha grado/norma. Cruza con el omega_gain que
  ya esta en results/merged.json (retencion omega_lib - retencion ewc).
  NO reentrena la tarea B ni evalua HumanEval: la estructura relevante es la
  de tras-A, y el gain ya lo tenemos medido.
=============================================================================
"""
import os, sys, json, argparse
import torch

ROOT = os.environ.get("OMEGA_ROOT", "/workspace/omega-s")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

# Reutilizamos la maquinaria EXACTA del experimento principal para que el
# entrenamiento de la tarea A sea identico al que produjo la tabla.
import rerun_retention as R

EPS = 1e-6


def components(W):
    """C, D, M, Coex de la matriz de peso efectiva W, con las MISMAS formulas
    que StochasticOmegaS (omega_s.py). Devuelve floats, sin gradiente."""
    with torch.no_grad():
        Wf = W.float()
        A = torch.sigmoid(torch.abs(Wf @ Wf.t()))
        A = 0.5 * (A + A.t())
        D = torch.mean(A).item()
        degrees = torch.sum(A, dim=1)
        Coex = (torch.var(degrees) + EPS).item()
        # Tr(A^3) exacto (matrices pequenas por modulo, no hace falta Hutchinson)
        A3 = A @ A @ A
        tr_a3 = torch.diagonal(A3).sum()
        C = (tr_a3 / (torch.norm(A, p="fro") ** 3 + EPS) + EPS).item()
        # Modularidad aproximada por una iteracion de potencia del laplaciano
        v = torch.randn(A.shape[0], 1, device=A.device)
        v = v / (torch.norm(v) + EPS)
        Lv = (degrees.view(-1, 1) * v) - (A @ v)
        M = (torch.abs((v.t() @ Lv).squeeze()) + EPS).item()
        # Brecha topologia-vs-normas: var de grados vs var de normas de fila
        row_norms = torch.norm(Wf, dim=1)
        var_rownorm = torch.var(row_norms).item()
        # correlacion grado-norma en ESTA matriz (la parte no explicada = topologico)
        dd = degrees - degrees.mean()
        rr = row_norms - row_norms.mean()
        denom = (torch.norm(dd) * torch.norm(rr) + EPS)
        corr_deg_norm = ((dd @ rr) / denom).item()
    return dict(C=C, D=D, M=M, Coex=Coex,
                var_rownorm=var_rownorm, corr_deg_norm=corr_deg_norm)


def measure_arm(seed, arm, best_wd=0.05, smoke=False):
    """Entrena SOLO la tarea A (codigo) con el brazo dado, replicando lo que
    hace run_cell para esa fase, y mide las componentes sobre los modulos LoRA
    justo al terminar A. NO entrena la tarea B ni evalua HumanEval."""
    print(f"\n=== semilla {seed} brazo {arm}: entrenando SOLO tarea A ===", flush=True)
    model, tok = R.load_model(seed, smoke)
    loader_A = R.get_loader(tok, "code", smoke)

    lam = None
    gen = torch.Generator().manual_seed(seed + 10_000)
    if arm.startswith("omega") or arm == "rownorm":
        R.assert_connected(model, arm, gen)
        b0 = next(iter(loader_A))
        lam = R.calibrate_lambda(model, {"input_ids": b0["input_ids"].to(R.DEVICE),
                                         "labels": b0["labels"].to(R.DEVICE)}, arm, gen)

    wd = best_wd if arm == "wd" else 0.0
    R.train_domain(model, loader_A, arm, wd, lam=lam, gen=gen)

    # medir componentes sobre cada modulo efectivo y promediar
    per_mod = [components(W) for _, W in R.iter_effective_weights(model)]
    keys = list(per_mod[0].keys())
    agg = {k: sum(m[k] for m in per_mod) / len(per_mod) for k in keys}
    del model
    torch.cuda.empty_cache()
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", default="both", choices=["none", "omega_lib", "both"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = ["none", "omega_lib"] if a.arm == "both" else [a.arm]
    rows = []
    for arm in arms:
        comp = measure_arm(a.seed, arm, smoke=a.smoke)
        comp.update(seed=a.seed, arm=arm)
        rows.append(comp)
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"  {arm}: " + "  ".join(f"{k}={v:.4g}" for k, v in comp.items()
                                        if k not in ("seed", "arm")), flush=True)
    print("Guardado en " + a.out, flush=True)
