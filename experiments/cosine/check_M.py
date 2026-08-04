# check_M.py  --  ¿puede moverse M bajo la construccion con sigmoid?
#
# El termino de modularidad M de StochasticOmegaS esta orientado al reves de lo
# que la Definicion 1 del FSRI requiere: la libreria estima lambda_2 y minimiza
# log(M*Coex/(C*D)), o sea empuja lambda_2 ABAJO, mientras que el indice con
# M = 1/lambda_2 necesita lambda_2 ARRIBA para ser alto.
#
# La pregunta que decide cuanto importa eso NO es cual es la orientacion
# correcta, sino si M se mueve siquiera. Bajo A = sigmoid(|.|) las afinidades
# colapsan al punto medio, el grafo queda casi constante, todos los grados casi
# iguales y lambda_2 casi fijado por el orden de la matriz y no por los pesos.
# Si es asi, M esta tan inerte como C y la orientacion no ha hecho nada en
# ninguna direccion.
#
# Se mide igual que se midio C: ELASTICIDAD, que es el cambio relativo de M
# ante un cambio relativo de W. Adimensional, asi que comparable entre modulos
# y entre factores.
#
#   elasticidad ~ 0    -> el factor esta muerto, no aporta gradiente
#   elasticidad ~ 1    -> responde proporcionalmente
#
# Referencia ya medida sobre estos mismos pesos:
#   C bajo sigmoid : mediana 0.84, MINIMO 0.00 (muerto en al menos un modulo)
#   C bajo coseno  : mediana 16.6
#
# Correr en ~/omega-sat:   python check_M.py

import glob
import json
import struct

import numpy as np

_snaps = glob.glob("/Users/acedo/.cache/huggingface/hub/"
                   "models--NousResearch--Meta-Llama-3-8B/snapshots/*")
if not _snaps:
    raise SystemExit("No encuentro el snapshot de Llama. Ajusta la ruta.")
SNAP = _snaps[0]
EPS = 1e-6


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
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
    raise KeyError(key)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def construir_A(W, rama):
    """Igual que StochasticOmegaS, incluida su bifurcacion por forma."""
    Wd = W.astype(np.float64)
    Wc = Wd if rama == "cuadrada" else Wd @ Wd.T
    S = sigmoid(np.abs(Wc))
    return (S + S.T) / 2.0


def factores(A):
    """C, D, M, Coex con las formulas de la libreria, TODOS de una sola A.
    M por lambda_2 exacto, para separar 'se mueve M?' de 'converge el
    estimador de la libreria?'."""
    k = A.sum(1)
    A2 = A @ A
    C = np.einsum("ij,ij->", A, A2) / (np.linalg.norm(A) ** 3 + EPS) + EPS
    ev = np.linalg.eigvalsh(np.diag(k) - A)
    return dict(C=C, D=A.mean(), M_lambda2=np.sort(ev)[1], Coex=k.var() + EPS)


# El espectro completo de una 4096x4096 cuesta minutos, y la pregunta ("se
# mueve M?") no necesita la matriz entera: el colapso del sigmoid al punto
# medio es un fenomeno por entrada, no de escala. Se submuestrean N_SUB filas,
# y se declara.
N_SUB = 1024


def elasticidades(W, rama, h=1e-3, semilla=0):
    """Las cuatro elasticidades de una vez: 3 construcciones de A por modulo
    en vez de 12. (dX/X)/(dW/W) en una direccion aleatoria fija."""
    rng = np.random.default_rng(semilla)
    if W.shape[0] > N_SUB:
        idx = rng.choice(W.shape[0], N_SUB, replace=False)
        W = W[idx][:, idx] if rama == "cuadrada" else W[idx]
    Dir = rng.standard_normal(W.shape).astype(np.float32)
    Dir /= np.linalg.norm(Dir)
    esc = h * np.linalg.norm(W)
    base = factores(construir_A(W, rama))
    mas = factores(construir_A(W + esc * Dir, rama))
    men = factores(construir_A(W - esc * Dir, rama))
    return {c: abs((mas[c] - men[c]) / (2 * h * base[c] + 1e-30)) for c in base}


CAPAS = [0, 8, 16, 24, 31]
print(f"snapshot: {SNAP}\n")
print("ELASTICIDAD de cada factor: cuanto se mueve ante un cambio de W.")
print(f"0 = muerto, no aporta gradiente.  Submuestreo a {N_SUB} filas.\n")
print(f"{'modulo':22} {'rama':>10} {'C':>10} {'D':>10} {'M(lam2)':>10} {'Coex':>10}")

filas = []
for L in CAPAS:
    for tipo in ("q_proj", "v_proj"):
        W = load_bf16(f"model.layers.{L}.self_attn.{tipo}.weight")
        rama = "cuadrada" if W.shape[0] == W.shape[1] else "gram"
        # v_proj es 1024x4096: el Gram sale 1024x1024, manejable.
        # q_proj es 4096x4096 y la rama cuadrada NO forma el Gram, tambien va.
        e = elasticidades(W, rama)
        print(f"L{L}.{tipo:16} {rama:>10} {e['C']:10.4f} {e['D']:10.4f} "
              f"{e['M_lambda2']:10.4f} {e['Coex']:10.4f}")
        filas.append(dict(capa=L, tipo=tipo, rama=rama, **e))

print("\n" + "=" * 78)
print("COMO LEERLO")
print("=" * 78)
med = {c: float(np.median([f[c] for f in filas]))
       for c in ("C", "D", "M_lambda2", "Coex")}
for c, v in med.items():
    print(f"  mediana de {c:10} {v:.4f}")
print()
if med["M_lambda2"] < 0.05:
    print("  M ESTA INERTE, igual que C. La orientacion invertida respecto a la")
    print("  Definicion 1 del FSRI no ha hecho nada, ni bueno ni malo, porque el")
    print("  factor no responde a los pesos bajo esta construccion. Lo que hay")
    print("  que corregir es la DESCRIPCION, no el comportamiento.")
elif med["M_lambda2"] < med["Coex"] / 10:
    print("  M responde, pero un orden de magnitud menos que Coex. La orientacion")
    print("  importa poco en la practica y conviene declararlo con esta cifra.")
else:
    print("  M SI RESPONDE de forma comparable a Coex. Entonces la orientacion")
    print("  invertida SI ha estado actuando, en contra de lo que el marco")
    print("  predice, y corregirla es un experimento con retorno esperado")
    print("  POSITIVO: una linea de codigo y diez celdas.")
print()
print("  Referencia sobre estos mismos pesos: C bajo sigmoid tiene mediana 0.84")
print("  con minimo 0.00, y bajo coseno 16.6.")
json.dump(filas, open("check_M.json", "w"), indent=2)
print("\nGuardado en check_M.json")
