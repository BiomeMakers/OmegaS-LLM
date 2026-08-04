#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_patch_2.py  --  brazo COMPUESTO con construccion coseno (`cos_composite`)

Es la replica EXACTA de StochasticOmegaS cambiando UNA SOLA COSA: como se
construye A. Todo lo demas (las cuatro formulas, el epsilon, el numero de
iteraciones de potencia, el log-ratio) es identico, para que la comparacion
contra `omega_lib` sea de una variable.

QUE SE COPIA TAL CUAL de omega_s.py:
    D      = mean(A)                          <- media de TODAS las entradas
    degrees= sum(A, dim=1)
    Coex   = var(degrees) + eps
    C      = Tr(A^3)_Hutchinson / (||A||_F^3 + eps) + eps
             OJO: esta es la normalizacion de la LIBRERIA (Frobenius al cubo),
             NO el clustering ponderado Tr(A^3)/(sum A^2 - Tr A^2) que se uso
             en los diagnosticos de la fase 1a. Son dos normalizaciones
             distintas y aqui manda la de la libreria.
    M      = |v^T L v| por iteracion de potencia, 3 iteraciones, v mean-centrado
    perdida= log((M * Coex) / (C * D + eps))

QUE CAMBIA:
    A = |cos(w_i, w_j)| por filas, diagonal a CERO, UNIFORME para todos los
    modulos.

    La libreria hace:
        if W.size(0) != W.size(1):  W_corr = W @ W.t()
        else:                       W_corr = W
    o sea que en Llama-3-8B q_proj y o_proj (4096x4096, cuadradas) reciben
    sigmoid(|W|) elemento a elemento, y k_proj y v_proj (1024x4096 por GQA)
    reciben sigmoid(|W W^T|). DOS construcciones distintas segun la forma.
    El coseno se aplica UNIFORME, lo que de paso arregla esa inconsistencia,
    pero implica que para q_proj la comparacion contra omega_lib tiene DOS
    diferencias en vez de una. Declararlo, o aislarlo con un brazo de control
    sigmoid_uniform si cos_composite acaba ganando.

DESVIACION MENOR DECLARADA: la libreria saca el vector inicial de la iteracion
de potencia con torch.randn del RNG GLOBAL; aqui se usa el generador dedicado
`gen`, igual que los demas brazos cos. Es mas limpio (no contamina el flujo
global) pero no es identico. Efecto de segundo orden frente a la varianza
entre semillas.

EL SIGNO NO ES LIBRE en el compuesto: viene fijado por la formula. Si se quiere
barrer, la forma limpia es un exponente sobre C, log((M*Coex)/(C^s * D)), que
se activa con COS_C_SIGN=-1. Por defecto +1 = comportamiento de la libreria.

Uso, desde la carpeta de rerun_retention.py:
    python apply_patch_2.py
"""
import ast
import os
import shutil
import sys

P = "rerun_retention.py"

CORE = '''
# ===========================================================================
# BRAZO COMPUESTO CON CONSTRUCCION COSENO
# Replica de StochasticOmegaS cambiando SOLO la construccion de A.
# ===========================================================================
OMEGA_EPS   = 1e-6
COS_C_SIGN  = float(os.environ.get("COS_C_SIGN", "1.0"))   # exponente sobre C


def cosine_composite(W, probes, gen):
    """log((M*Coex)/(C^s*D)) con A construida por coseno por filas.

    Las cuatro formulas son las de omega_s.py palabra por palabra; lo unico
    distinto es A. Devuelve un escalar conectado al grafo.
    """
    eps = OMEGA_EPS
    A = cosine_A(W, zero_diag=True)          # <- LA UNICA DIFERENCIA
    N = A.size(0)

    D = torch.mean(A)
    degrees = torch.sum(A, dim=1)
    Coex = torch.var(degrees) + eps

    # Hutchinson para Tr(A^3), normalizado por Frobenius al cubo (libreria)
    tr_A3 = 0.0
    for _ in range(probes):
        z = (torch.randint(0, 2, (N, 1), generator=gen,
                           dtype=torch.float32) * 2.0 - 1.0).to(A.device)
        tr_A3 = tr_A3 + torch.matmul(
            z.t(), torch.matmul(A, torch.matmul(A, torch.matmul(A, z)))).squeeze()
    tr_A3 = tr_A3 / probes
    C = tr_A3 / (torch.norm(A, p="fro") ** 3 + eps) + eps

    # Modularidad espectral por iteracion de potencia sobre el laplaciano
    v = torch.randn((N, 1), generator=gen, dtype=torch.float32).to(A.device)
    v = v - torch.mean(v)
    v = v / (torch.norm(v) + eps)
    max_deg = torch.max(degrees)
    for _ in range(3):
        Lv = (degrees.view(-1, 1) * v) - torch.matmul(A, v)
        v = ((2 * max_deg) * v - Lv)
        v = (v - torch.mean(v)) / (torch.norm(v) + eps)
    M_est = torch.abs(torch.matmul(
        v.t(), (degrees.view(-1, 1) * v) - torch.matmul(A, v)).squeeze()) + eps

    # C^s: s=+1 reproduce la libreria (minimizar SUBE C). s=-1 invierte solo
    # la direccion del canal de clustering, dejando M, Coex y D como estan.
    return torch.log((M_est * Coex) / ((C ** COS_C_SIGN) * D + eps))

'''

PEN_OLD = '''    elif arm in ("cos_full", "cos_noCoex"):'''
PEN_NEW = '''    elif arm == "cos_composite":
        total = sum(cosine_composite(mods[i][1], OMEGA_PROBES, gen) for i in idx)
    elif arm in ("cos_full", "cos_noCoex"):'''

ARMS_OLD = '"omega_lib", "cos_full", "cos_noCoex"]'
ARMS_NEW = '"omega_lib", "cos_full", "cos_noCoex", "cos_composite"]'

PHASE2 = '''
# ===========================================================================
# FASE 2: el compuesto coseno contra omega_lib, sobre las MISMAS 10 semillas
# ===========================================================================
def run_phase2(smoke, out, he_n, cell_idx=None):
    """Celdas = SEEDS x P2_ARMS, en el orden fijo de SEEDS.

    Por defecto solo cos_composite (10 celdas), porque las columnas de
    none/wd/ewc/omega_lib ya existen en la tabla de 10 semillas y el montaje
    es el mismo (q/v, target 0.03, EVERY_K=10). Si se quiere re-correr algun
    incumbente, pasarlo en P2_ARMS.

    cell_idx reparte entre GPUs con CUDA_VISIBLE_DEVICES, una celda por proceso.
    """
    import os as _os
    arms = _os.environ.get("P2_ARMS", "cos_composite").split(",")
    _os.environ.setdefault("OMEGA_TARGET", "0.03")
    CELLS = [(sd, ar) for sd in SEEDS for ar in arms]
    idxs = [cell_idx] if cell_idx is not None else list(range(len(CELLS)))

    print("\\n" + "#" * 66)
    print("FASE 2: compuesto coseno frente a omega_lib (0.7662, 10 semillas)")
    print("  brazos=" + str(arms) + "  target=" + _os.environ["OMEGA_TARGET"] +
          "  C^s con s=" + str(COS_C_SIGN))
    print("  celdas de este proceso: " + str(idxs) + " de " + str(len(CELLS)))
    print("#" * 66)

    rows = []
    for i in idxs:
        sd, ar = CELLS[i]
        print("\\n### celda " + str(i) + ": " + ar + "  semilla " + str(sd) + " ###")
        r = run_cell(sd, ar, 0.0, smoke, he_n)
        r["cell_idx"] = i
        rows.append(r)
        json.dump(rows, open(out, "w"), indent=2)
    print("\\nGuardado en " + out)
    return rows

'''

ARG_OLD = '''    ap.add_argument("--p1b-cell", type=int, default=None,
                    help="indice de celda 0-3 para repartir 1b entre GPUs")'''
ARG_NEW = '''    ap.add_argument("--p1b-cell", type=int, default=None,
                    help="indice de celda 0-3 para repartir 1b entre GPUs")
    ap.add_argument("--phase2", action="store_true",
                    help="fase 2: compuesto coseno sobre las 10 semillas")
    ap.add_argument("--p2-cell", type=int, default=None,
                    help="indice de celda para repartir la fase 2 entre GPUs")'''

DISP_OLD = '''    if a.phase1b:
        run_phase1b(a.smoke, a.out, he_n, a.p1b_cell)
        return'''
DISP_NEW = '''    if a.phase1b:
        run_phase1b(a.smoke, a.out, he_n, a.p1b_cell)
        return

    if a.phase2:
        run_phase2(a.smoke, a.out, he_n, a.p2_cell)
        return'''


def main():
    if not os.path.exists(P):
        sys.exit("No encuentro " + P + " aqui.")
    src = open(P).read()

    if "def cosine_A(" not in src:
        sys.exit("ABORTO: falta el parche de 1a (no existe cosine_A).")
    if "def run_phase1b(" not in src:
        sys.exit("ABORTO: falta el parche de 1b.")

    if 'cos_composite"]' in src:
        print("ARMS ya estaba")
    else:
        if ARMS_OLD not in src:
            sys.exit("ABORTO: no encuentro la lista ARMS con los brazos cos")
        src = src.replace(ARMS_OLD, ARMS_NEW, 1)
        print("ARMS: anadido cos_composite")

    if "def cosine_composite(" in src:
        print("nucleo compuesto ya estaba")
    else:
        anchor = "\ndef cosine_tr_a3_hutch("
        if anchor not in src:
            sys.exit("ABORTO: no encuentro cosine_tr_a3_hutch para anclar")
        src = src.replace(anchor, CORE + anchor, 1)
        print("nucleo cosine_composite insertado")

    if 'arm == "cos_composite"' in src:
        print("dispatch en omega_pen ya estaba")
    else:
        if PEN_OLD not in src:
            sys.exit("ABORTO: no encuentro la rama cos de omega_pen")
        src = src.replace(PEN_OLD, PEN_NEW, 1)
        print("omega_pen: enruta cos_composite")

    if "def run_phase2(" in src:
        print("bloque fase 2 ya estaba")
    else:
        anchor = "\ndef main():"
        if anchor not in src:
            sys.exit("ABORTO: no encuentro def main()")
        src = src.replace(anchor, PHASE2 + anchor, 1)
        print("bloque run_phase2 insertado")

    if "--phase2" in src:
        print("flags fase 2 ya estaban")
    else:
        if ARG_OLD not in src:
            sys.exit("ABORTO: no encuentro el flag --p1b-cell")
        src = src.replace(ARG_OLD, ARG_NEW, 1)
        print("flags --phase2 y --p2-cell anadidos")

    if "if a.phase2:" in src:
        print("dispatch fase 2 ya estaba")
    else:
        if DISP_OLD not in src:
            sys.exit("ABORTO: no encuentro el dispatch de --phase1b")
        src = src.replace(DISP_OLD, DISP_NEW, 1)
        print("dispatch de la fase 2 anadido")

    try:
        ast.parse(src)
    except SyntaxError as e:
        sys.exit("ABORTO: el resultado no compila: " + str(e))

    if not os.path.exists(P + ".pre2"):
        shutil.copy(P, P + ".pre2")
        print("copia de seguridad: " + P + ".pre2")
    open(P, "w").write(src)
    print("\nOK. " + P + " parcheado con el compuesto coseno y verificado.")
    print("\nSmoke (1 celda, HumanEval reducido, ~10 min):")
    print("  cd /workspace/omega-s")
    print("  HF_HOME=/workspace/hf OMEGA_EVERY_K=10 python \\\\")
    print("    experiments/rerun_retention.py --phase2 --p2-cell 0 \\\\")
    print("    --he-n 32 --out p2_smoke.json")
    print("\nFase 2 completa, 10 celdas en 8 GPUs (dos tandas):")
    print("  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i \\\\")
    print("    HF_HOME=/workspace/hf OMEGA_EVERY_K=10 nohup python \\\\")
    print("    experiments/rerun_retention.py --phase2 --p2-cell $i \\\\")
    print("    --out p2_cell$i.json > p2_$i.log 2>&1 & done")
    print("  # y despues las celdas 8 y 9")


if __name__ == "__main__":
    main()
