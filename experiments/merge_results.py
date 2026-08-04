#!/usr/bin/env python3
"""
Fusiona los resultados de las GPUs y aplica la regla de decision prerregistrada.

Uso:
    python merge_results.py results/
"""
import sys, json, glob, os
import statistics as st

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results"
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        rows += json.load(open(f))
    if not rows:
        print("Sin resultados en " + d); return

    json.dump(rows, open(os.path.join(d, "merged.json"), "w"), indent=2)
    print(str(len(rows)) + " celdas fusionadas -> " + d + "/merged.json\n")

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {})[r["seed"]] = r

    print(f"{'brazo':<12} {'retencion media':>16} {'desv':>8} {'plasticidad':>13} {'n':>3}")
    print("-" * 58)
    for arm in ["none", "wd", "ewc", "rownorm", "omega_raw", "omega_lib"]:
        cells = by_arm.get(arm, {})
        if not cells: continue
        rets = [c["retention_pct"] for c in cells.values()]
        plas = [c["humaneval_after_B"] for c in cells.values()]
        sd = st.stdev(rets) if len(rets) > 1 else 0.0
        print(f"{arm:<12} {st.mean(rets):>16.4f} {sd:>8.4f} "
              f"{st.mean(plas):>13.4f} {len(rets):>3}")

    print("\n" + "=" * 58)
    print("REGLA DE DECISION PRERREGISTRADA")
    print("=" * 58)

    wd = by_arm.get("wd", {})
    ewc = by_arm.get("ewc", {})
    if not wd:
        print("Falta el brazo wd. No se puede evaluar."); return
    ewc_rets = [c["retention_pct"] for c in ewc.values()] if ewc else []
    ewc_mean = st.mean(ewc_rets) if ewc_rets else float("nan")
    ewc_sd = st.stdev(ewc_rets) if len(ewc_rets) > 1 else 0.0

    for arm in ["omega_raw", "omega_lib"]:
        cells = by_arm.get(arm, {})
        if not cells:
            print(f"\n{arm}: sin datos"); continue
        seeds = sorted(set(cells) & set(wd))
        wins = sum(1 for s in seeds
                   if cells[s]["retention_pct"] > wd[s]["retention_pct"]
                   and cells[s]["humaneval_after_B"] >= wd[s]["humaneval_after_B"] - 0.01)
        rets = [cells[s]["retention_pct"] for s in seeds]
        mean = st.mean(rets) if rets else float("nan")
        ok_ewc = (mean >= ewc_mean - ewc_sd) if ewc_rets else None

        print(f"\n{arm}")
        print(f"  (a)+(b) bate a wd ajustado sin perder plasticidad: {wins}/{len(seeds)} semillas")
        print(f"  (c) vs EWC: {mean:.4f} vs {ewc_mean:.4f} +/- {ewc_sd:.4f}"
              f"  -> {'cumple' if ok_ewc else 'NO cumple'}")
        veredicto = (wins >= 2) and bool(ok_ewc)
        print(f"  VEREDICTO: {'VALIDADO' if veredicto else 'NO VALIDADO'}")
        if not veredicto:
            print("             El claim de retencion NO se publica para este metodo.")

    # Control de normas de fila: omega_lib debe superar a rownorm
    rn = by_arm.get("rownorm", {})
    olib = by_arm.get("omega_lib", {})
    if rn and olib:
        seeds = sorted(set(rn) & set(olib))
        wins = sum(1 for s in seeds
                   if olib[s]["retention_pct"] > rn[s]["retention_pct"])
        rn_rets = [rn[s]["retention_pct"] for s in seeds]
        ol_rets = [olib[s]["retention_pct"] for s in seeds]
        gap = st.mean(ol_rets) - st.mean(rn_rets)
        rn_sd = st.stdev(rn_rets) if len(rn_rets) > 1 else 0.0
        print("\n" + "=" * 58)
        print("CONTROL DE NORMAS DE FILA: omega_lib vs rownorm")
        print("=" * 58)
        print(f"  omega_lib gana en {wins}/{len(seeds)} semillas")
        print(f"  gap medio {gap:+.4f}  (desv rownorm {rn_sd:.4f})")
        if wins > len(seeds) * 0.6 and gap > rn_sd:
            print("  >>> omega_lib SUPERA al control. Hace algo mas que igualar normas.")
        elif abs(gap) <= rn_sd:
            print("  >>> EMPATE. El mecanismo ES igualar normas de fila.")
        else:
            print("  >>> Mixto. Mirar semilla a semilla.")

    print("\nRecordatorio: estos numeros SUSTITUYEN al 83.03/81.07 del README y")
    print("del preprint. Los antiguos salieron de corridas con la penalizacion")
    print("desconectada del grafo. Ver AUDIT.md.")


if __name__ == "__main__":
    main()
