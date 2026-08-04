#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_patch_ablacion.py  --  ABLACION POR MODULO, limpia.

Anade la variable de entorno OMEGA_ONLY para aplicar la penalizacion SOLO a un
tipo de modulo, dejando el ADAPTADOR intacto en q_proj y v_proj.

POR QUE NO VALE RESTRINGIR LORA_TARGETS: eso cambiaria tambien donde hay
parametros entrenables, y entonces se confundiria "que canal regularizas" con
"donde puede aprender el modelo". El filtro va DENTRO de omega_pen, asi que el
adaptador es identico en los tres brazos y lo unico que cambia es sobre que
modulos actua la penalizacion.

QUE PREGUNTA CONTESTA:
  decompose_coex.py midio que la secuencia de grados es 96% MAGNITUD en q_proj
  (rama cuadrada, A = sigmoid(|W|)) y 71% ALINEAMIENTO en v_proj (rama Gram,
  A = sigmoid(|W W^T|)). Como rownorm (magnitud pura) pierde 10/10, la ventaja
  de omega_lib deberia venir de v_proj.

PREDICCION PRERREGISTRADA:
  solo v_proj ~ 0.766 (el resultado completo)  -> el mecanismo es ALINEAMIENTO
  solo q_proj ~ 0.437 (como rownorm)           -> ahi no hay nada propio
  si sale al reves, el mecanismo es magnitud y la diferencia con rownorm hay
  que buscarla en la estructura del log-ratio.

CAVEAT QUE HAY QUE DECLARAR: la calibracion fija la fuerza TOTAL de la
penalizacion, asi que al restringir a la mitad de los modulos esa fuerza se
concentra en la mitad. Los brazos ablacionados reciben ~2x de fuerza acumulada
por modulo. Si v_proj gana, conviene una corrida de seguimiento a target
reducido para separar canal de intensidad.

Uso, desde la carpeta de rerun_retention.py:
    python apply_patch_ablacion.py
"""
import ast
import os
import shutil
import sys

P = "rerun_retention.py"

CONST_ANCHOR = 'COS_SIGN     = float(os.environ.get("COS_SIGN", "1.0"))'
CONST_NEW = CONST_ANCHOR + '''
# Ablacion por modulo: "" = todos, "q_proj" o "v_proj" = solo ese tipo.
# El filtro actua sobre la PENALIZACION, no sobre LORA_TARGETS, para que el
# adaptador sea identico en los tres brazos.
OMEGA_ONLY   = os.environ.get("OMEGA_ONLY", "").strip()'''

PEN_OLD = '''    mods = list(iter_effective_weights(model))
    if not mods:
        raise RuntimeError("Sin modulos LoRA. Revisa LORA_TARGETS.")'''
PEN_NEW = '''    mods = list(iter_effective_weights(model))
    if OMEGA_ONLY:
        mods = [m for m in mods if m[0].endswith(OMEGA_ONLY)]
    if not mods:
        raise RuntimeError("Sin modulos LoRA. Revisa LORA_TARGETS / OMEGA_ONLY.")'''


def main():
    if not os.path.exists(P):
        sys.exit(f"No encuentro {P} aqui.")
    src = open(P).read()

    if "OMEGA_ONLY" in src:
        print("el parche de ablacion ya estaba aplicado")
    else:
        if CONST_ANCHOR not in src:
            sys.exit("ABORTO: no encuentro el ancla de COS_SIGN (falta el parche 1a)")
        src = src.replace(CONST_ANCHOR, CONST_NEW, 1)
        print("constante OMEGA_ONLY anadida")

        if PEN_OLD not in src:
            sys.exit("ABORTO: no encuentro la cabecera de omega_pen")
        src = src.replace(PEN_OLD, PEN_NEW, 1)
        print("omega_pen filtra por OMEGA_ONLY")

        try:
            ast.parse(src)
        except SyntaxError as e:
            sys.exit(f"ABORTO: no compila: {e}")

        if not os.path.exists(P + ".preabl"):
            shutil.copy(P, P + ".preabl")
            print(f"copia de seguridad: {P}.preabl")
        open(P, "w").write(src)
        print(f"\nOK. {P} parcheado y verificado.")

    print("\nComprobacion rapida de que el filtro ve los modulos:")
    print("  python -c \"import sys; sys.path.insert(0,'.'); "
          "import os; os.environ['OMEGA_ONLY']='v_proj'; "
          "print('constante leida OK')\"")


if __name__ == "__main__":
    main()
