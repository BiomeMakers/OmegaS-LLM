#!/usr/bin/env python3
"""Lee la fase de ajuste y saca el mejor hiperparametro de cada brazo.
Uso: python merge_tuning.py results_tuning/"""
import sys, json, glob, os, statistics as st

d = sys.argv[1] if len(sys.argv) > 1 else "results_tuning"
rows = []
for f in glob.glob(os.path.join(d, "*.json")):
    try: rows += json.load(open(f))
    except Exception: pass
if not rows:
    print("Sin resultados en", d); sys.exit()

json.dump(rows, open(os.path.join(d, "tuning_merged.json"), "w"), indent=2)
print(len(rows), "celdas de ajuste\n")

# clave del hiperparametro segun brazo
def hp(r):
    if r["arm"] == "wd": return r["wd"]
    if r["arm"] == "ewc": return r.get("ewc_lambda", r.get("omega_lambda"))
    return r.get("omega_target")

# agrupa por (brazo, valor) y promedia sobre semillas
agg = {}
for r in rows:
    agg.setdefault((r["arm"], hp(r)), []).append(r["retention_pct"])

print(f"{'brazo':<11}{'valor':>10}{'retencion':>12}{'desv':>9}{'n':>4}")
print("-"*46)
best = {}
for (arm, val), rets in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
    m = st.mean(rets); sd = st.stdev(rets) if len(rets) > 1 else 0.0
    print(f"{arm:<11}{str(val):>10}{m:>12.4f}{sd:>9.4f}{len(rets):>4}")
    if arm not in best or m > best[arm][1]:
        best[arm] = (val, m)

print("\n" + "="*46)
print("MEJOR HIPERPARAMETRO DE CADA BRAZO")
print("="*46)
for arm in ["wd", "ewc", "omega_raw", "omega_lib"]:
    if arm in best:
        val, m = best[arm]
        print(f"  {arm:<11} valor optimo = {val}   (retencion {m:.4f})")

print("\nEstos son los valores para la corrida de EVALUACION final.")
print("Fijalos y corre las semillas reservadas con cada brazo en su optimo.")
