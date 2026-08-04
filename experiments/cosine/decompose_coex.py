# decompose_coex.py  --  ¿de qué está hecha Coex?
#
# Coex = var(grados) es el UNICO canal por el que opera omega_lib (dC ~ 0 en
# 10/10). La pregunta que queda abierta es de qué esta hecha esa varianza:
#
#   canal de MAGNITUD    : la parte de k_i que predice la norma de la fila
#   canal de ALINEAMIENTO: el residuo, o sea cuanto se alinea esa fila con las
#                          demas mas alla de lo que explica su tamano
#
# IMPORTA porque hoy se midio que `rownorm` (igualar normas de fila) no solo no
# ayuda: PERJUDICA (0.437 frente a 0.766, 10/10, p=0.002). Si Coex fuese casi
# toda magnitud, omega_lib y rownorm harian casi lo mismo y eso seria una
# contradiccion. Si Coex es sobre todo alineamiento, la contradiccion se
# disuelve y el mecanismo tiene nombre.
#
# Se calculan las DOS ramas, porque la libreria bifurca segun la forma:
#   q_proj (4096x4096, cuadrada) -> A = sigmoid(|W|)
#   v_proj (1024x4096)           -> A = sigmoid(|W W^T|)
#
# LIMITACION DECLARADA: esto es sobre pesos BASE. No dice cual de los dos
# canales se MUEVE durante el entrenamiento; eso exige adaptadores guardados y
# ninguna corrida los guardo. Es la mitad estatica, la que puede falsar la
# hipotesis barata.
#
# Correr en ~/omega-sat:   python decompose_coex.py

import numpy as np
import glob
import struct
import json

_snaps = glob.glob("/Users/acedo/.cache/huggingface/hub/"
                   "models--NousResearch--Meta-Llama-3-8B/snapshots/*")
if not _snaps:
    raise SystemExit("No encuentro el snapshot. Ajusta la ruta.")
SNAP = _snaps[0]


def load_bf16(key):
    for path in sorted(glob.glob(f"{SNAP}/*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            if key not in hdr:
                continue
            meta = hdr[key]
            s, e = meta["data_offsets"]
            f.seek(8 + n + s)
            raw = f.read(e - s)
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(meta["shape"])
    raise KeyError(key)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def grados(W, rama):
    """Secuencia de grados tal como la calcula StochasticOmegaS."""
    Wd = W.astype(np.float64)
    if rama == "cuadrada":
        A = sigmoid(np.abs(Wd))
    else:
        A = sigmoid(np.abs(Wd @ Wd.T))
    A = (A + A.T) / 2.0
    return A.sum(axis=1)


def descomponer(k, norma):
    """Reparte var(k) entre lo que predice la norma de fila y el residuo.

    Regresion lineal simple k ~ a + b*norma. R2 es la fraccion de la varianza
    de grados que el canal de MAGNITUD explica; 1-R2 es la del canal de
    ALINEAMIENTO.
    """
    x = (norma - norma.mean()) / (norma.std() + 1e-12)
    y = k - k.mean()
    b = float((x * y).sum() / ((x * x).sum() + 1e-12))
    pred = b * x
    resid = y - pred
    r2 = 1.0 - resid.var() / (y.var() + 1e-12)
    return dict(coex=float(y.var()), r2=float(r2),
                var_mag=float(pred.var()), var_ali=float(resid.var()),
                corr_norma_resid=float(np.corrcoef(norma, resid)[0, 1]))


CAPAS = [0, 8, 16, 24, 31]
print(f"snapshot: {SNAP}\n")
print("Coex = var(grados).  R2 = fraccion que explica la NORMA de fila.")
print("El resto (1-R2) es el canal de ALINEAMIENTO.\n")

for tipo, rama in (("q_proj", "cuadrada"), ("v_proj", "gram")):
    print("=" * 86)
    print(f"{tipo}   rama que usa la libreria: {rama}")
    print("=" * 86)
    print(f"{'capa':>5} {'n':>6} {'Coex':>12} {'R2 norma':>10} "
          f"{'% magnitud':>11} {'% alineam.':>11} {'corr(norma,res)':>16}")
    for L in CAPAS:
        W = load_bf16(f"model.layers.{L}.self_attn.{tipo}.weight")
        k = grados(W, rama)
        nrm = np.linalg.norm(W.astype(np.float64), axis=1)
        d = descomponer(k, nrm)
        print(f"{L:5d} {W.shape[0]:6d} {d['coex']:12.4e} {d['r2']:10.4f} "
              f"{100*d['r2']:10.1f}% {100*(1-d['r2']):10.1f}% "
              f"{d['corr_norma_resid']:16.4f}")
    print()

print("=" * 86)
print("COMO LEERLO")
print("=" * 86)
print("  R2 ALTO (>0.7): Coex es sobre todo MAGNITUD. Entonces omega_lib y")
print("    rownorm actuan sobre casi lo mismo, y que uno ayude y el otro")
print("    perjudique NO se explica por el canal: habria que buscar la causa")
print("    en la estructura del log-ratio. La hipotesis del alineamiento CAE.")
print("  R2 BAJO (<0.3): Coex es sobre todo ALINEAMIENTO. La paradoja se")
print("    disuelve: rownorm empuja la magnitud y omega_lib el alineamiento,")
print("    que son cosas distintas. El mecanismo pasa a tener nombre.")
print("  Y mira corr(norma, residuo): si es NEGATIVA, magnitud y alineamiento")
print("    van en sentidos opuestos, que es lo que ya se midio en la fase 1")
print("    (las filas de norma grande apuntan a direcciones idiosincrasicas y")
print("    por eso se alinean menos). Eso explicaria que rownorm no sea un")
print("    Omega debil sino algo que empuja casi al reves.")
