# verify_cosine.py  --  comprobar C/D ~ 1.357 con coseno + diagonal a cero
#
# Version sin index.json: escanea el header de cada shard y localiza el
# tensor por si mismo. Correr en ~/omega-sat con el venv.
#
#   python verify_cosine.py

import numpy as np
import glob
import struct
import json

from sat_test_v4 import c_cosine   # reusa TU funcion exacta

_snaps = glob.glob(
    "/Users/acedo/.cache/huggingface/hub/"
    "models--NousResearch--Meta-Llama-3-8B/snapshots/*")
if not _snaps:
    raise SystemExit("No encuentro el snapshot. Ajusta SNAP a mano.")
SNAP = _snaps[0]


def read_header(path):
    """Devuelve (dict_header, offset_de_datos) de un .safetensors."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def find_shard(key):
    """Busca en que shard esta 'key' leyendo los headers."""
    for path in sorted(glob.glob(f"{SNAP}/*.safetensors")):
        hdr, base = read_header(path)
        if key in hdr:
            return path, hdr[key], base
    raise KeyError(f"{key} no esta en ningun shard")


def load_bf16(path, meta, base):
    """Lee el tensor bf16 y lo pasa a fp32 (gotcha safetensors+numpy)."""
    s, e = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(meta["shape"])


def excess_CD(A):
    A = A.astype(np.float64)
    n = A.shape[0]
    A2 = A @ A
    C = np.einsum("ij,ij->", A, A2) / (A2.sum() - np.trace(A2) + 1e-12)
    off = A.sum() - np.trace(A)
    D = off / (n * (n - 1) + 1e-12)
    return C / (D + 1e-12)


targets = [
    "model.layers.0.self_attn.q_proj.weight",   # n=4096
    "model.layers.0.self_attn.o_proj.weight",   # n=4096
    "model.layers.0.self_attn.k_proj.weight",   # n=1024 (GQA)
]

print(f"snapshot: {SNAP}\n")
for key in targets:
    path, meta, base = find_shard(key)
    W = load_bf16(path, meta, base)
    A = c_cosine(W)
    shard = path.split("/")[-1]
    print(f"{key:48s}  n={W.shape[0]:5d}  C/D = {excess_CD(A):.4f}  [{shard}]")

print("\nSi q/o rondan 1.3-1.4 y k > 1, verificado: se entrena el objeto medido.")
