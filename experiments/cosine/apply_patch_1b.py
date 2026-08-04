#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_patch_1b.py  --  FASE 1b sobre rerun_retention.py (ya parcheado con 1a).

1b contesta lo que dC NO puede contestar: QUE SIGNO AYUDA A RETENER. Eso exige
HumanEval, asi que va por la ruta normal de run_cell y no por train_phase1a.

Diseno: 2 signos x 2 semillas FUERA DE MUESTRA (7077, 8088) a target fijo,
retencion = HumanEval(tras B) / HumanEval(tras A). 4 celdas.

Contiene TRES arreglos, uno de ellos un fallo que habria petado en el arranque:
  [FIX 1] CRITICO. run_cell solo calibra lambda si el brazo empieza por "omega"
          o es "rownorm". Los brazos cos se quedaban con lam=None y, como el
          edit 6 de 1a SI les aplica la penalizacion en train_domain, la
          ejecucion moria en  None * omega_pen(...)  -> TypeError.
  [FIX 2] omega_target volvia None en la fila de resultados para los brazos cos.
  [FIX 3] LORA_TARGETS vuelve a q_proj/v_proj.

Por que volver a q/v, y son tres razones a la vez:
  (a) preserva el pareado con las columnas guardadas de las 10 semillas, que
      se corrieron con q/v; con q/k/v/o el montaje es otro y el pareado no vale
  (b) v_proj resulto ser el respondedor MAS FUERTE en 1a-bis (3.17% a target
      0.03); solo se pierden k (1.5%) y o (0.07%)
  (c) 64 modulos en vez de 128 = la mitad de dilucion por muestreo

Uso, desde la carpeta de rerun_retention.py:
    python apply_patch_1b.py
"""
import ast
import os
import shutil
import sys

P = "rerun_retention.py"

# ---------------------------------------------------------------- FIX 1
CAL_OLD = '''    lam = None
    if arm.startswith("omega") or arm == "rownorm":
        assert_connected(model, arm, gen)'''
CAL_NEW = '''    lam = None
    if arm.startswith("omega") or arm == "rownorm" or arm.startswith("cos"):
        assert_connected(model, arm, gen)'''

# ---------------------------------------------------------------- FIX 2
RET_OLD = '''    return dict(seed=seed, arm=arm, wd=wd, omega_lambda=lam,
                omega_target=float(os.environ.get("OMEGA_TARGET", "0.1"))
                             if (arm.startswith("omega") or arm=="rownorm") else None,'''
RET_NEW = '''    return dict(seed=seed, arm=arm, wd=wd, omega_lambda=lam,
                cos_sign=COS_SIGN if arm.startswith("cos") else None,
                omega_target=float(os.environ.get("OMEGA_TARGET", "0.1"))
                             if (arm.startswith("omega") or arm=="rownorm"
                                 or arm.startswith("cos")) else None,'''

# ---------------------------------------------------------------- bloque 1b
PHASE1B = '''
# ===========================================================================
# FASE 1b: que SIGNO ayuda a retener. Con HumanEval, no con dC.
# ===========================================================================
def run_phase1b(smoke, out, he_n, cell_idx=None):
    """2 signos x 2 semillas fuera de muestra, target fijo, retencion medida.

    Las semillas son NUEVAS (no estan en SEEDS) a proposito: el ajuste anterior
    uso 42 y 123, que si estan entre las diez de evaluacion, y eso produjo el
    artefacto none > wd por regresion a la media. Ajustar fuera de muestra
    handicapa al brazo nuevo, que es el sesgo conservador que interesa.

    cell_idx permite lanzar UNA celda por proceso y repartir entre GPUs con
    CUDA_VISIBLE_DEVICES: load_model hace .to("cuda") sobre un solo device, asi
    que 4 procesos en 4 GPUs corren las 4 celdas en paralelo.
    """
    import os as _os
    seeds = [int(x) for x in _os.environ.get("P1B_SEEDS", "7077,8088").split(",")]
    target = _os.environ.get("P1B_TARGET", "0.03")
    _os.environ["OMEGA_TARGET"] = target
    arm = _os.environ.get("P1B_ARM", "cos_full")

    CELLS = [(sg, sd) for sg in (1.0, -1.0) for sd in seeds]
    idxs = [cell_idx] if cell_idx is not None else list(range(len(CELLS)))

    print("\\n" + "#" * 66)
    print("FASE 1b: que direccion del termino de clustering AYUDA a retener")
    print("  brazo=" + arm + "  target=" + target + "  semillas=" + str(seeds))
    print("  signo +1 BAJA C  |  signo -1 SUBE C (direccion del FSRI)")
    print("  celdas de este proceso: " + str(idxs) + " de " + str(len(CELLS)))
    print("#" * 66)

    rows = []
    for i in idxs:
        sg, sd = CELLS[i]
        _set_cos_sign(sg)
        print("\\n### celda " + str(i) + ": signo " + format(sg, "+.0f") +
              "  semilla " + str(sd) + " ###")
        r = run_cell(sd, arm, 0.0, smoke, he_n)
        r["cell_idx"] = i
        rows.append(r)
        json.dump(rows, open(out, "w"), indent=2)

    # resumen solo si este proceso corrio todas las celdas
    if cell_idx is None and len(rows) == len(CELLS):
        print("\\n" + "=" * 66)
        print("FASE 1b: RETENCION POR SIGNO Y SEMILLA")
        for r in rows:
            print("  signo " + format(r["cos_sign"], "+.0f") + "  semilla " +
                  str(r["seed"]) + "  retencion " +
                  format(r["retention_pct"], ".4f") +
                  "  (A=" + format(r["humaneval_after_A"], ".4f") +
                  " B=" + format(r["humaneval_after_B"], ".4f") + ")")
        print("-" * 66)
        # comparacion PAREADA por semilla: es el estadistico con potencia
        pares = []
        for sd in seeds:
            up = [r for r in rows if r["seed"] == sd and r["cos_sign"] < 0]
            dn = [r for r in rows if r["seed"] == sd and r["cos_sign"] > 0]
            if up and dn:
                d = up[0]["retention_pct"] - dn[0]["retention_pct"]
                pares.append(d)
                print("  semilla " + str(sd) + ": subir C menos bajar C = " +
                      format(d, "+.4f"))
        if pares:
            n_up = sum(1 for d in pares if d > 0)
            print("-" * 66)
            print("  subir C gana en " + str(n_up) + " de " + str(len(pares)) +
                  " semillas; diferencia media " +
                  format(sum(pares) / len(pares), "+.4f"))
            print("\\n  LECTURA, y con 2 semillas es CRIBADO y no confirmacion:")
            print("  si gana subir C (signo -1), la direccion del FSRI se")
            print("  sostiene en LLM y la transferencia ecologica es real.")
            print("  Si gana bajar C (signo +1), la direccion del indice")
            print("  depende del dominio y hay que escribirlo en el FSRI.")
            print("  Si empatan, el canal se mueve pero no toca la retencion")
            print("  (desenlace 2) y lo que decide es el brazo compuesto.")
        print("=" * 66)
    print("\\nGuardado en " + out)
    return rows

'''

ARG_OLD = '''    ap.add_argument("--phase1a", action="store_true",
                    help="cribado de vitalidad coseno: targets x signos, con puerta")'''
ARG_NEW = '''    ap.add_argument("--phase1a", action="store_true",
                    help="cribado de vitalidad coseno: targets x signos, con puerta")
    ap.add_argument("--phase1b", action="store_true",
                    help="1b: que signo ayuda a retener, con HumanEval")
    ap.add_argument("--p1b-cell", type=int, default=None,
                    help="indice de celda 0-3 para repartir 1b entre GPUs")'''

DISP_OLD = '''    if a.phase1a:
        run_phase1a(a.smoke, a.out)
        return'''
DISP_NEW = '''    if a.phase1a:
        run_phase1a(a.smoke, a.out)
        return

    if a.phase1b:
        run_phase1b(a.smoke, a.out, he_n, a.p1b_cell)
        return'''


def main():
    if not os.path.exists(P):
        sys.exit("No encuentro " + P + " aqui.")
    src = open(P).read()

    if "def run_phase1a(" not in src:
        sys.exit("ABORTO: este fichero no tiene el parche de 1a. Aplica "
                 "apply_patch_1a.py primero.")

    # FIX 3: LORA_TARGETS de vuelta a q/v
    if 'LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]' in src:
        src = src.replace('LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]',
                          'LORA_TARGETS = ["q_proj", "v_proj"]  # 1b/fase2: pareado con las 10 semillas')
        print("FIX 3 aplicado: LORA_TARGETS de vuelta a q_proj/v_proj")
    else:
        print("FIX 3 ya estaba (o el fichero no tiene q/k/v/o)")

    # FIX 1: el que habria petado.
    # OJO con la guarda: buscar solo 'or arm.startswith("cos")' da FALSO
    # POSITIVO, porque esa cadena ya existe en train_domain por el edit 6 de
    # 1a. Hay que comprobar el bloque COMPLETO de calibracion de run_cell.
    if CAL_NEW in src:
        print("FIX 1 ya estaba")
    else:
        if CAL_OLD not in src:
            sys.exit("ABORTO: no encuentro el bloque de calibracion de run_cell")
        src = src.replace(CAL_OLD, CAL_NEW, 1)
        print("FIX 1 aplicado: run_cell calibra lambda para los brazos cos "
              "(sin esto: None * omega_pen -> TypeError)")

    # FIX 2
    if "cos_sign=COS_SIGN" in src:
        print("FIX 2 ya estaba")
    else:
        if RET_OLD not in src:
            sys.exit("ABORTO: no encuentro el dict de retorno de run_cell")
        src = src.replace(RET_OLD, RET_NEW, 1)
        print("FIX 2 aplicado: la fila de resultados guarda cos_sign y omega_target")

    # bloque 1b
    if "def run_phase1b(" in src:
        print("bloque 1b ya estaba")
    else:
        anchor = "\ndef main():"
        if anchor not in src:
            sys.exit("ABORTO: no encuentro def main()")
        src = src.replace(anchor, PHASE1B + anchor, 1)
        print("bloque run_phase1b insertado")

    # flags y dispatch
    if "--phase1b" in src:
        print("flags 1b ya estaban")
    else:
        if ARG_OLD not in src:
            sys.exit("ABORTO: no encuentro el flag --phase1a")
        src = src.replace(ARG_OLD, ARG_NEW, 1)
        print("flags --phase1b y --p1b-cell anadidos")

    if "if a.phase1b:" in src:
        print("dispatch 1b ya estaba")
    else:
        if DISP_OLD not in src:
            sys.exit("ABORTO: no encuentro el dispatch de --phase1a")
        src = src.replace(DISP_OLD, DISP_NEW, 1)
        print("dispatch de 1b anadido")

    try:
        ast.parse(src)
    except SyntaxError as e:
        sys.exit("ABORTO: el resultado no compila: " + str(e))

    if not os.path.exists(P + ".pre1b"):
        shutil.copy(P, P + ".pre1b")
        print("copia de seguridad: " + P + ".pre1b")
    open(P, "w").write(src)
    print("\nOK. " + P + " parcheado para 1b y verificado.")
    print("\nLanzamiento en PARALELO, 4 celdas en 4 GPUs (~1 h en vez de ~4):")
    print("  cd /workspace/omega-s")
    print("  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i HF_HOME=/workspace/hf \\")
    print("    OMEGA_EVERY_K=1 P1B_TARGET=0.03 nohup python experiments/rerun_retention.py \\")
    print("    --phase1b --p1b-cell $i --out p1b_cell$i.json > p1b_$i.log 2>&1 & done")


if __name__ == "__main__":
    main()
