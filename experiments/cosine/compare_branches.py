# compare_branches.py  --  ¿es material la rama cuadrado/no cuadrado?
#
# omega_s.py hace:
#     if W.size(0) != W.size(1):  W_corr = W @ W.t()
#     else:                       W_corr = W
#     A = sigmoid(|W_corr|)
# En Llama-3-8B eso significa que q_proj y o_proj (4096x4096, CUADRADAS)
# reciben sigmoid(|W|) elemento a elemento, y k_proj y v_proj (1024x4096 por
# GQA) reciben sigmoid(|W W^T|). El paper describe solo la segunda.
#
# Este script calcula los CUATRO factores del indice (C, D, Coex, M) sobre
# pesos REALES bajo las tres construcciones, para decidir si hace falta un
# brazo de control sigmoid_uniform o si la rama es inmaterial.
#
# Coste cero de GPU. Correr en ~/omega-sat con el venv:
#     python compare_branches.py

import numpy as np
import glob
import struct
import json

_snaps = glob.glob(
    "/Users/acedo/.cache/huggingface/hub/"
    "models--NousResearch--Meta-Llama-3-8B/snapshots/*")
if not _snaps:
    raise SystemExit("No encuentro el snapshot. Ajusta la ruta a mano.")
SNAP = _snaps[0]
EPS = 1e-6


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def load_bf16(key):
    for path in sorted(glob.glob(f"{SNAP}/*.safetensors")):
        hdr, base = read_header(path)
        if key in hdr:
            meta = hdr[key]
            s, e = meta["data_offsets"]
            with open(path, "rb") as f:
                f.seek(base + s)
                raw = f.read(e - s)
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.reshape(meta["shape"])
    raise KeyError(key)


# ---------------- las tres construcciones ----------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def A_square_branch(W):
    """Lo que hace la libreria si W es CUADRADA: sigmoid(|W|), sin Gram."""
    A = sigmoid(np.abs(W.astype(np.float64)))
    return (A + A.T) / 2.0


def A_gram_branch(W):
    """Lo que hace la libreria si NO es cuadrada, y lo que dice el paper."""
    G = W.astype(np.float64) @ W.astype(np.float64).T
    A = sigmoid(np.abs(G))
    return (A + A.T) / 2.0


def A_cosine(W, eps=1e-8):
    """La construccion nueva: |cos| por filas, diagonal a cero."""
    Wf = W.astype(np.float64)
    Wn = Wf / (np.linalg.norm(Wf, axis=1, keepdims=True) + eps)
    A = np.clip(np.abs(Wn @ Wn.T), 0.0, 1.0)
    np.fill_diagonal(A, 0.0)
    return A


# ---------------- los cuatro factores, formulas de la libreria ----------------
def factors(A, n_power=3, seed=0):
    N = A.shape[0]
    D = A.mean()
    degrees = A.sum(axis=1)
    Coex = degrees.var() + EPS

    # C = Tr(A^3) / ||A||_F^3   (normalizacion de la LIBRERIA)
    trA3 = np.einsum("ij,ij->", A, A @ A)
    C = trA3 / (np.linalg.norm(A) ** 3 + EPS) + EPS

    # M por iteracion de potencia sobre el laplaciano, igual que la libreria
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 1))
    v = v - v.mean()
    v = v / (np.linalg.norm(v) + EPS)
    max_deg = degrees.max()
    for _ in range(n_power):
        Lv = degrees.reshape(-1, 1) * v - A @ v
        v = (2 * max_deg) * v - Lv
        v = (v - v.mean()) / (np.linalg.norm(v) + EPS)
    M = abs(float((v.T @ (degrees.reshape(-1, 1) * v - A @ v)).squeeze())) + EPS

    omega = np.log((M * Coex) / (C * D + EPS))
    return dict(C=C, D=D, Coex=Coex, M=M, omega=omega,
                A_mean=A.mean(), A_std=A.std())


targets = [
    ("q_proj  (4096x4096, CUADRADA)", "model.layers.0.self_attn.q_proj.weight"),
    ("v_proj  (1024x4096, no cuadrada)", "model.layers.0.self_attn.v_proj.weight"),
]

print(f"snapshot: {SNAP}\n")
for label, key in targets:
    W = load_bf16(key)
    print("=" * 78)
    print(f"{label}   shape={W.shape}")
    print("=" * 78)
    builds = [("sigmoid(|W|)  rama cuadrada", A_square_branch),
              ("sigmoid(|WW^T|) rama Gram  ", A_gram_branch),
              ("coseno, diag 0             ", A_cosine)]
    if W.shape[0] != W.shape[1]:
        builds = builds[1:]          # la rama cuadrada no aplica
    print(f"{'construccion':30} {'C':>10} {'D':>10} {'Coex':>12} "
          f"{'M':>12} {'omega':>10}")
    for name, fn in builds:
        A = fn(W)
        f = factors(A)
        print(f"{name:30} {f['C']:10.6f} {f['D']:10.6f} {f['Coex']:12.4e} "
              f"{f['M']:12.4e} {f['omega']:10.4f}")
    print()

print("COMO LEERLO: si en q_proj las dos ramas del sigmoid dan los cuatro")
print("factores casi iguales, la rama es INMATERIAL y no hace falta el brazo")
print("de control sigmoid_uniform. Si divergen, el control se justifica.")
print("La fila del coseno es la construccion nueva, para ver cuanto se separa")
print("de las dos.")
