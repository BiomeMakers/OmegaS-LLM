# =============================================================================
# AÑADIR WEIGHT DECAY A LA EVALUACION, sin repetir el resto
#
# Reutiliza lo que ya esta en la sesion: los scripts en disco y exp6b.json.
# Son ~20 corridas (5 valores de rejilla x 2 semillas de ajuste, mas 10 de
# evaluacion). Al final fusiona el resultado dentro de exp6b.json.
#
# CRITERIO, el de exp6: se elige el lambda de WD que IGUALA la accuracy del
# modelo completo del brazo de referencia, no la de `none`. Comparar WD en su
# mejor accuracy contra Omega en la suya no es una comparacion.
# =============================================================================
import json
import torch

from exp6_wd_vs_omega_pruning import (
    DEVICE, SEED, get_loader, build_model, train_wd, eval_acc,
    run_pruning_pipeline,
)

MAX_CAIDA  = 1.0                                   # pp de accuracy que puede perder la poda
TUNE_SEEDS = [7077, 8088]
EVAL_SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
# rejilla EXTENDIDA: la de exp6 llega a 1e-2 y ahi la accuracy aun ronda 96.5%,
# asi que para igualar ~96.1% hace falta bajar mas
WD_GRID_EXT = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

d = json.load(open("/content/exp6b.json"))
tr, te = get_loader(train=True), get_loader(train=False)

# referencia: accuracy del modelo COMPLETO del brazo de referencia
ref_arm = "gl+raw"
ref = next(r["acc_media"] for r in d["brazos"] if r["brazo"] == ref_arm)
print(f"objetivo de accuracy (modelo completo de {ref_arm}): {ref:.4%}\n")


def medir(model, etiqueta):
    acc = eval_acc(model, te)
    pruned, ff, fp = run_pruning_pipeline(model, etiqueta)
    acc_p = eval_acc(pruned, te)
    red = 1.0 - fp / ff
    caida = 100 * (acc - acc_p)
    return dict(acc=acc, acc_podado=acc_p, caida_pp=caida, reduccion=red,
                valido=bool(caida < MAX_CAIDA),
                flops_full=int(ff), flops_pruned=int(fp))


# ---------------------------------------------------------------- AJUSTE
print("AJUSTE de WD sobre semillas fuera de muestra")
cand = []
for lam in WD_GRID_EXT:
    accs = []
    for sd in TUNE_SEEDS:
        torch.manual_seed(sd)
        m = train_wd(build_model(), tr, lam)
        accs.append(eval_acc(m, te))
    a = sum(accs) / len(accs)
    print(f"  WD {lam:.0e} -> acc {a:.4%}   |dif con la referencia| {abs(a-ref):.4%}")
    cand.append((lam, a))

best_lam, best_acc = min(cand, key=lambda c: abs(c[1] - ref))
print(f"\n  -> elegido WD {best_lam:.0e} (acc {best_acc:.4%}, "
      f"referencia {ref:.4%})\n")

# ------------------------------------------------------------ EVALUACION
print(f"EVALUACION de WD {best_lam:.0e} sobre las {len(EVAL_SEEDS)} semillas")
rs = []
for sd in EVAL_SEEDS:
    torch.manual_seed(sd)
    m = train_wd(build_model(), tr, best_lam)
    r = medir(m, f"wd@s{sd}")
    rs.append(r)
    print(f"  s{sd:<5} acc {r['acc']:.4%} -> {r['acc_podado']:.4%}"
          f"  FLOPs -{r['reduccion']:.2%}  {'OK' if r['valido'] else 'ROTO'}")

fila = dict(brazo=f"wd({best_lam:.0e})", config=dict(wd=best_lam), por_semilla=rs,
            media=sum(r["reduccion"] for r in rs) / len(rs),
            n_validas=sum(1 for r in rs if r["valido"]),
            acc_media=sum(r["acc"] for r in rs) / len(rs),
            accp_media=sum(r["acc_podado"] for r in rs) / len(rs))
d["brazos"] = [b for b in d["brazos"] if not b["brazo"].startswith("wd(")] + [fila]
json.dump(d, open("/content/exp6b.json", "w"), indent=2)

# ---------------------------------------------------------------- LECTURA
print("\n" + "=" * 78)
print("TABLA FINAL, restringida a las semillas donde la poda NO rompe el modelo")
print("=" * 78)
print(f"{'brazo':14} {'validas':>9} {'FLOPs':>10} {'acc completo':>14} {'acc podado':>12}")
filas = []
for r in d["brazos"]:
    v = [p for p in r["por_semilla"] if p["valido"]]
    if not v:
        print(f"  {r['brazo']:14} {'0/10':>9}   sin semillas validas")
        continue
    f = sum(p["reduccion"] for p in v) / len(v)
    a1 = sum(p["acc"] for p in v) / len(v)
    a2 = sum(p["acc_podado"] for p in v) / len(v)
    filas.append((r["brazo"], len(v), f, a1, a2))
for b, n, f, a1, a2 in sorted(filas, key=lambda x: (-x[1], -x[2])):
    print(f"  {b:14} {n:>6}/10 {f:>9.2%} {a1:>14.4%} {a2:>12.4%}")

print("\n" + "=" * 78)
print("COMO LEERLO")
print("=" * 78)
print("  Dos ejes, y hay que mirar los dos:")
print("    FIABILIDAD  = en cuantas semillas la poda no rompe el modelo")
print("    COMPRESION  = cuantos FLOPs se quitan cuando si funciona")
print()
print("  Si WD sale 10/10 con menos FLOPs -> es un CANJE, no una ventaja limpia:")
print("     Omega comprime mas pero WD nunca falla.")
print("  Si WD sale con pocas validas Y menos FLOPs -> Omega gana en los dos ejes.")
print("  Si WD sale 10/10 con FLOPs parecidos -> el resultado de poda no es de")
print("     Omega y hay que reescribir esa seccion del paper.")
