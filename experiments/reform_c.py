#!/usr/bin/env python3
"""
=============================================================================
REFORMULACION DE C CON CONTRASTE PRESERVADO + TEST DE SATURACION
=============================================================================

Contexto: la medicion de mecanismo (measure_structure) mostro que el termino
de clustering C esta INERTE porque A = sigmoid(|W W^T|) comprime todo a
[0.5,1), y el clustering normalizado Tr(A^3)/||A||^3 se clava en ~0.9997.

Esta reformulacion (vía temperatura, la mas simple del apendice del preprint)
construye A con contraste:  A_tau = sigmoid( |W W^T| / tau  -  b )
con b = mediana de |W W^T|/tau  (asi ~mitad de las entradas caen bajo 0.5).

TEST DE SATURACION (que hace este fichero cuando se ejecuta directo):
  mide C con la formula ORIGINAL y con la REFORMULADA sobre los pesos ya
  entrenados de una semilla, y comprueba si C deja de estar clavado en 0.9997.
  Barre tau para elegir el que da MAS rango dinamico de C.
  Si C no se descomprime, NO tiene sentido entrenar: se aborta.
=============================================================================
"""
import os, sys, json, argparse
import torch

ROOT = os.environ.get("OMEGA_ROOT", "/workspace/omega-s")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import rerun_retention as R

EPS = 1e-6


def affinity_original(W):
    Wf = W.float()
    A = torch.sigmoid(torch.abs(Wf @ Wf.t()))
    return 0.5 * (A + A.t())


def affinity_reformed(W, tau, b=None):
    """A_tau = sigmoid(|WW^T|/tau - b), b = mediana si no se da."""
    Wf = W.float()
    G = torch.abs(Wf @ Wf.t()) / tau
    if b is None:
        b = torch.median(G)
    A = torch.sigmoid(G - b)
    return 0.5 * (A + A.t())


def affinity_degree_norm(W):
    """Normalizacion por grado: A_norm = D^{-1/2} A D^{-1/2} sobre A=sigmoid(|WW^T|).
    Es la construccion que la literatura de clusterabilidad usa (Filan et al.
    2021, sobre el laplaciano normalizado) para que el clustering no lo dominen
    las magnitudes/grados. Mejor fundamentada que el rango."""
    Wf = W.float()
    A = torch.sigmoid(torch.abs(Wf @ Wf.t()))
    A = 0.5 * (A + A.t())
    deg = A.sum(dim=1)
    dinv = torch.rsqrt(deg + EPS)
    A_norm = dinv.view(-1, 1) * A * dinv.view(1, -1)
    return A_norm


def affinity_rank(W, max_elems=200_000):
    """Via iii del apendice: mapea |WW^T| por su CDF empirica a [0,1], via
    interpolacion contra una SUBMUESTRA de valores (ordenar la matriz entera
    -4096x4096 = 16M por modulo- agota la RAM). Ignora magnitudes: C dependeria
    solo del ORDEN de las afinidades = topologia pura. Diagonal excluida del
    muestreo porque domina (||fila||^2)."""
    Wf = W.float()
    G = torch.abs(Wf @ Wf.t())
    n = G.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=G.device)
    vals = G[off]
    # submuestra para estimar la CDF sin ordenar los 16M
    m = vals.numel()
    if m > max_elems:
        idx = torch.randperm(m, device=G.device)[:max_elems]
        sample = vals[idx]
    else:
        sample = vals
    sample_sorted, _ = torch.sort(sample)
    # CDF empirica: rank de cada valor = posicion en la muestra ordenada
    # searchsorted da, para cada g, cuantos valores de la muestra son <= g
    ranks = torch.searchsorted(sample_sorted, G.reshape(-1)).float()
    A = (ranks / (sample_sorted.numel() + EPS)).reshape(n, n)
    A = 0.5 * (A + A.t())
    A.fill_diagonal_(1.0)
    return A


def clustering_C(A):
    """Tr(A^3)/||A||_F^3, exacto (matrices por modulo son pequenas)."""
    A3 = A @ A @ A
    tr = torch.diagonal(A3).sum()
    return (tr / (torch.norm(A, p="fro") ** 3 + EPS) + EPS).item()


def clustering_C_hutch(A, n, gen):
    """Version Hutchinson, la que usaria el entrenamiento (insesgada en grad)."""
    N = A.shape[0]; total = 0.0
    for _ in range(n):
        z = (torch.randint(0, 2, (N, 1), generator=gen, dtype=torch.float32,
                           device=A.device) * 2 - 1)
        total = total + (z.t() @ (A @ (A @ (A @ z)))).squeeze()
    tr = total / n
    return (tr / (torch.norm(A, p="fro") ** 3 + EPS) + EPS).item()


def run_test(seed, taus, smoke=False):
    """Entrena SOLO tarea A (brazo none) y mide C original vs reformada."""
    print(f"=== TEST DE SATURACION, semilla {seed} ===", flush=True)
    model, tok = R.load_model(seed, smoke)
    loader = R.get_loader(tok, "code", smoke)
    R.train_domain(model, loader, "none", 0.0)

    mods = [W for _, W in R.iter_effective_weights(model)]
    print(f"modulos LoRA: {len(mods)}")

    # C original
    C_orig = sum(clustering_C(affinity_original(W)) for W in mods) / len(mods)
    print(f"\nC ORIGINAL (sigmoid|WW^T|): {C_orig:.6f}  "
          f"{'<-- SATURADO' if C_orig > 0.99 else ''}")

    # C reformada, barriendo tau
    print("\nC REFORMADA por temperatura (buscamos rango dinamico ancho):")
    best = None
    results = {}
    for tau in taus:
        Cs = [clustering_C(affinity_reformed(W, tau)) for W in mods]
        Cmean = sum(Cs) / len(Cs)
        Cspread = max(Cs) - min(Cs)  # rango entre modulos = contraste recuperado
        results[tau] = dict(mean=Cmean, spread=Cspread)
        flag = ""
        if Cmean < 0.95:  # se ha despegado del techo
            flag = "  <-- descomprimido"
        print(f"  tau={tau:<8g} C_mean={Cmean:.4f}  spread_entre_modulos={Cspread:.4f}{flag}")
        if best is None or Cspread > results[best]["spread"]:
            best = tau

    print(f"\nMejor tau (mas contraste): {best}  "
          f"(C_mean={results[best]['mean']:.4f}, spread={results[best]['spread']:.4f})")

    # Via de RANGO (iii): la que ignora magnitudes
    print("\nC REFORMADA por RANGO (CDF empirica, ignora magnitudes):")
    Cs_rank = [clustering_C(affinity_rank(W)) for W in mods]
    Crank_mean = sum(Cs_rank) / len(Cs_rank)
    Crank_spread = max(Cs_rank) - min(Cs_rank)
    rank_ok = Crank_mean < 0.95 and Crank_spread > 0.01
    print(f"  rango: C_mean={Crank_mean:.4f}  spread={Crank_spread:.4f}"
          f"{'  <-- DESCOMPRIMIDO' if rank_ok else ''}")

    # Via de NORMALIZACION POR GRADO: la respaldada por bibliografia
    print("\nC REFORMADA por GRADO (D^-1/2 A D^-1/2, Filan et al. 2021):")
    Cs_deg = [clustering_C(affinity_degree_norm(W)) for W in mods]
    Cdeg_mean = sum(Cs_deg) / len(Cs_deg)
    Cdeg_spread = max(Cs_deg) - min(Cs_deg)
    deg_ok = Cdeg_mean < 0.95 and Cdeg_spread > 0.01
    print(f"  grado: C_mean={Cdeg_mean:.4f}  spread={Cdeg_spread:.4f}"
          f"{'  <-- DESCOMPRIMIDO' if deg_ok else ''}")

    temp_ok = results[best]["mean"] < 0.95 and results[best]["spread"] > 0.01
    descomprime = temp_ok or rank_ok or deg_ok
    # preferencia: grado (bibliografia) > rango > temperatura
    if deg_ok:      metodo = "grado (D^-1/2 A D^-1/2, respaldo bibliografico)"
    elif rank_ok:   metodo = "rango"
    elif temp_ok:   metodo = "temperatura"
    else:           metodo = "ninguno"
    print("\n" + "=" * 60)
    if descomprime:
        print(f">>> FUNCIONA por via '{metodo}'. C se descomprime.")
        print("    Recomendada la via de grado si descomprime (tiene cita).")
        print("    Siguiente: entrenar clustering_only con esa via, 5 semillas.")
    else:
        print(">>> NINGUNA via descomprime C (temperatura, rango, ni grado).")
        print("    La saturacion es estructural, no solo de escala. NO entrenar.")
        print("    Se reporta como hallazgo: restaurar el canal topologico")
        print("    requiere replantear la construccion de A, no solo re-mapearla.")
    print("=" * 60)

    del model; torch.cuda.empty_cache()
    return dict(seed=seed, C_orig=C_orig, best_tau=best,
                C_rank_mean=Crank_mean, C_rank_spread=Crank_spread,
                C_deg_mean=Cdeg_mean, C_deg_spread=Cdeg_spread,
                metodo=metodo, descomprime=descomprime,
                results={str(k): v for k, v in results.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--taus", default="0.1,0.5,1,2,5,10,20,50")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results_reform/saturation_test.json")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = run_test(a.seed, taus, a.smoke)
    json.dump(res, open(a.out, "w"), indent=2)
    print("\nGuardado en " + a.out)
