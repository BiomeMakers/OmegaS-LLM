"""check_M.py -- which factors of the Omega-S objective actually respond to the weights.

This is the one measurement the paper asks others to run (Section 3.5). It needs
no training, no previous-task data and no GPU: it reads the weights of an
already-downloaded model and measures, for each of the four factors of the
objective, its ELASTICITY with respect to the weights.

Elasticity is the relative change in a factor produced by a relative change in W
along a fixed random direction. It is dimensionless, so it is comparable across
factors, modules and models:

    elasticity ~ 0    the factor is inert; it cannot contribute gradient
    elasticity ~ 1    the factor responds proportionally

On Llama-3-8B we measure C, D and M at or below 1e-4 and Coex around 9e-3, which
is what the paper means when it says three of the four factors are inert and the
effect is carried by degree variance. Whether that holds on other models is
open, and it is the single thing that would most change what the paper can say.

A report that the factors are LIVE on your model is as useful to us as one that
confirms they are not.

Usage:

    python experiments/check_M.py                          # default: Llama-3-8B
    MODEL=Qwen/Qwen2.5-7B python experiments/check_M.py
    MODEL=/path/to/a/local/snapshot python experiments/check_M.py

The model must already be present in the local Hugging Face cache (or given as a
path). Nothing is downloaded. Results are written to check_M.json; please attach
that file to the replication report.
"""

import glob
import json
import os
import struct
import sys

import numpy as np

EPS = 1e-6

# The full spectrum of a 4096x4096 matrix costs minutes, and the question ("does
# the factor move?") does not need the whole matrix: the collapse of the logistic
# towards its midpoint is a per-entry phenomenon, not a question of scale. We
# subsample N_SUB rows and declare it.
N_SUB = 1024

# How many layers to sample, spread evenly through the depth of the model.
N_LAYERS_SAMPLED = 5


# --------------------------------------------------------------------------
# Locating the weights
# --------------------------------------------------------------------------

def resolve_snapshot(model):
    """Return a directory containing .safetensors files for `model`.

    Accepts either a local path or a Hugging Face repo id already present in the
    local cache. Nothing is downloaded.
    """
    if os.path.isdir(model) and glob.glob(os.path.join(model, "*.safetensors")):
        return model

    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    pattern = os.path.join(cache, "models--" + model.replace("/", "--"), "snapshots", "*")
    snaps = [d for d in glob.glob(pattern) if glob.glob(os.path.join(d, "*.safetensors"))]
    if snaps:
        return sorted(snaps)[-1]

    sys.exit(
        f"Could not find weights for {model!r}.\n"
        f"Looked in: {pattern}\n\n"
        "Either download it first (huggingface-cli download <repo-id>), or pass a\n"
        "local directory that contains .safetensors files:\n"
        "    MODEL=/path/to/snapshot python experiments/check_M.py"
    )


def read_headers(snapshot):
    """Map every tensor key to (file, metadata) across all shards."""
    index = {}
    for path in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for key, meta in hdr.items():
            if key != "__metadata__":
                index[key] = (path, meta, n)
    return index


def load_tensor(index, key):
    """Load one 2-D tensor as float32, converting from bf16 or fp16 if needed."""
    path, meta, n = index[key]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + n + start)
        raw = f.read(end - start)
    dtype = meta["dtype"]
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dtype in ("F16", "FP16"):
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype in ("F32", "FP32"):
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        sys.exit(f"Unsupported dtype {dtype} for {key}. Please open an issue.")
    return arr.reshape(meta["shape"])


def find_projections(index):
    """Find the attention query/value projections and the layers they live in.

    Works with the common naming schemes (model.layers.N.self_attn.q_proj.weight
    and friends). Returns a list of (layer, label, key), sampled through depth.
    """
    found = {}
    for key in index:
        if not key.endswith(".weight"):
            continue
        parts = key.split(".")
        layer = next((int(p) for p in parts if p.isdigit()), None)
        if layer is None:
            continue
        for label in ("q_proj", "v_proj", "q_lin", "v_lin", "query", "value"):
            if label in parts:
                found.setdefault(layer, {})[label] = key
    if not found:
        sys.exit(
            "Could not find attention projections in this checkpoint.\n"
            "Tensor names look like: " + ", ".join(list(index)[:3]) + "\n"
            "Please open an issue with those names and we will add the pattern."
        )

    layers = sorted(found)
    step = max(1, (len(layers) - 1) // max(1, N_LAYERS_SAMPLED - 1))
    sampled = layers[::step][:N_LAYERS_SAMPLED]
    if layers[-1] not in sampled:
        sampled[-1] = layers[-1]

    out = []
    for layer in sampled:
        for label, key in sorted(found[layer].items()):
            out.append((layer, label, key))
    return out


# --------------------------------------------------------------------------
# The four factors, exactly as the library computes them
# --------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_A(W, branch):
    """As in StochasticOmegaS, including its branch on matrix shape."""
    Wd = W.astype(np.float64)
    Wc = Wd if branch == "square" else Wd @ Wd.T
    S = sigmoid(np.abs(Wc))
    return (S + S.T) / 2.0


def factors(A):
    """C, D, M, Coex with the library's formulas, all from one A.

    M is the exact lambda_2, so that "does M move?" is separated from "does the
    library's estimator converge?".
    """
    k = A.sum(1)
    A2 = A @ A
    C = np.einsum("ij,ij->", A, A2) / (np.linalg.norm(A) ** 3 + EPS) + EPS
    ev = np.linalg.eigvalsh(np.diag(k) - A)
    return dict(C=C, D=A.mean(), M_lambda2=np.sort(ev)[1], Coex=k.var() + EPS)


def elasticities(W, branch, h=1e-3, seed=0):
    """All four elasticities at once: (dX/X)/(dW/W) along a fixed random direction."""
    rng = np.random.default_rng(seed)
    if W.shape[0] > N_SUB:
        idx = rng.choice(W.shape[0], N_SUB, replace=False)
        W = W[idx][:, idx] if branch == "square" else W[idx]
    direction = rng.standard_normal(W.shape).astype(np.float32)
    direction /= np.linalg.norm(direction)
    scale = h * np.linalg.norm(W)
    base = factors(build_A(W, branch))
    up = factors(build_A(W + scale * direction, branch))
    down = factors(build_A(W - scale * direction, branch))
    return {c: abs((up[c] - down[c]) / (2 * h * base[c] + 1e-30)) for c in base}


# --------------------------------------------------------------------------

def main():
    model = os.environ.get("MODEL", "NousResearch/Meta-Llama-3-8B")
    is_reference_model = "llama-3-8b" in model.lower().replace("meta-", "")
    snapshot = resolve_snapshot(model)
    index = read_headers(snapshot)
    projections = find_projections(index)

    print(f"model:    {model}")
    print(f"snapshot: {snapshot}\n")
    print("ELASTICITY of each factor: how much it moves when W moves.")
    print(f"0 = inert, contributes no gradient. Subsampled to {N_SUB} rows.\n")
    print(f"{'module':24} {'branch':>8} {'C':>10} {'D':>10} {'M(lam2)':>10} {'Coex':>10}")

    rows = []
    for layer, label, key in projections:
        W = load_tensor(index, key)
        if W.ndim != 2:
            continue
        branch = "square" if W.shape[0] == W.shape[1] else "gram"
        e = elasticities(W, branch)
        name = f"L{layer}.{label}"
        print(f"{name:24} {branch:>8} {e['C']:10.4f} {e['D']:10.4f} "
              f"{e['M_lambda2']:10.4f} {e['Coex']:10.4f}")
        rows.append(dict(layer=layer, module=label, branch=branch, **e))

    if not rows:
        sys.exit("No 2-D projections were measured. Please open an issue.")

    print("\n" + "=" * 78)
    print("HOW TO READ IT")
    print("=" * 78)
    median = {c: float(np.median([r[c] for r in rows]))
              for c in ("C", "D", "M_lambda2", "Coex")}
    for c, v in median.items():
        print(f"  median {c:10} {v:.4f}")
    print()

    if median["Coex"] <= 0:
        verdict = "degenerate"
        print("  Coex does not move either. Something is wrong with the reading;")
        print("  please open an issue with check_M.json attached.")
    elif median["M_lambda2"] < 0.05 and median["C"] < 0.05:
        verdict = "inert"
        print("  C AND M ARE INERT here. Under this construction the objective")
        print("  reduces in practice to a penalty on the variance of node degrees.")
        if is_reference_model:
            print("  This is the model the paper measures, so it is a check that the")
            print("  script agrees with Table 3 rather than new evidence.")
        else:
            print("  This REPRODUCES the paper's finding on a model we did not")
            print("  measure, which is a useful result and we would like to hear")
            print("  about it.")
    elif median["M_lambda2"] < median["Coex"] / 10:
        verdict = "weak"
        print("  The factors respond, but an order of magnitude less than Coex.")
        print("  Degree variance still dominates, though less completely than on")
        print("  Llama-3-8B. Worth reporting with the numbers above.")
    else:
        verdict = "live"
        print("  THE FACTORS ARE LIVE here, comparably to Coex. This does")
        print("  NOT reproduce our finding, and it is the more interesting outcome:")
        print("  it would mean the reduction to degree variance is specific to the")
        print("  models we measured rather than general. Please report it.")

    print()
    print("  Reference on Llama-3-8B base weights: C, D and M at or below 1e-4,")
    print("  Coex around 9e-3.")

    out = dict(model=model, n_sub=N_SUB, verdict=verdict, median=median, rows=rows)
    with open("check_M.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWritten to check_M.json")
    print("Report: https://github.com/BiomeMakers/OmegaS-LLM/issues/new?template=replication.yml")


if __name__ == "__main__":
    main()
