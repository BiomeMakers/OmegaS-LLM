#!/usr/bin/env python3
"""
Analisis del experimento de estructura. Cruza las componentes medidas
(measure_structure) con el omega_gain de results/merged.json, y aplica el
criterio de falsacion PRERREGISTRADO en measure_structure.py.

Uso: python analyze_structure.py results_structure/ results/merged.json
"""
import sys, json, glob, os
import statistics as st
from collections import defaultdict


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), n
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan"), n
    return num / (dx * dy), n


def approx_p(r, n):
    """p aproximado (two-sided) via t. Solo orientativo con n pequeno."""
    if n < 3 or abs(r) >= 1:
        return float("nan")
    import math
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # aproximacion normal de la cola (suficiente para n~10-15, orientativo)
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return p


def main():
    struct_dir = sys.argv[1] if len(sys.argv) > 1 else "results_structure"
    merged = sys.argv[2] if len(sys.argv) > 2 else "results/merged.json"

    # 1. cargar componentes por semilla y brazo
    comp = defaultdict(dict)  # comp[seed][arm] = {C,D,M,Coex,...}
    for f in glob.glob(os.path.join(struct_dir, "*.json")):
        try:
            for r in json.load(open(f)):
                comp[r["seed"]][r["arm"]] = r
        except Exception:
            pass

    # 2. cargar gain por semilla desde merged.json
    rows = json.load(open(merged))
    ret = defaultdict(dict)
    for r in rows:
        ret[r["seed"]][r["arm"]] = r["retention_pct"]
    gain_vs_ewc = {s: ret[s]["omega_lib"] - ret[s]["ewc"]
                   for s in ret if "omega_lib" in ret[s] and "ewc" in ret[s]}
    gain_vs_wd = {s: ret[s]["omega_lib"] - ret[s]["wd"]
                  for s in ret if "omega_lib" in ret[s] and "wd" in ret[s]}

    seeds = sorted(s for s in comp if "none" in comp[s] and "omega_lib" in comp[s]
                   and s in gain_vs_ewc)
    print(f"Semillas con datos completos: {len(seeds)}  {seeds}\n")
    if len(seeds) < 3:
        print("Insuficientes para correlacionar. Faltan mediciones.")
        return

    # 3. construir las variables estructurales por semilla
    #    principales: dC = C_omega - C_none  (subida de clustering)
    #                 dCoex = Coex_none - Coex_omega (bajada de var grados)
    def col(fn):
        return [fn(comp[s]) for s in seeds]

    dC     = col(lambda c: c["omega_lib"]["C"] - c["none"]["C"])
    dCoex  = col(lambda c: c["none"]["Coex"] - c["omega_lib"]["Coex"])
    dD     = col(lambda c: c["omega_lib"]["D"] - c["none"]["D"])
    dM     = col(lambda c: c["none"]["M"] - c["omega_lib"]["M"])
    # estructura de PARTIDA (brazo none): brecha grado-norma
    corr_dn_none = col(lambda c: c["none"]["corr_deg_norm"])
    gap_none     = col(lambda c: 1.0 - abs(c["none"]["corr_deg_norm"]))  # 1-|corr| = parte no explicada por normas

    g_ewc = [gain_vs_ewc[s] for s in seeds]
    g_wd  = [gain_vs_wd[s] for s in seeds]

    N = 5  # numero de metricas -> Bonferroni
    thr = 0.05 / N

    print("=" * 66)
    print("CORRELACIONES CON gain_vs_ewc  (Bonferroni: p<%.3f para robusto)" % thr)
    print("=" * 66)
    tests = [
        ("PRINCIPAL  dC (sube clustering, TOPOLOGICO)", dC),
        ("RIVAL      dCoex (baja var grados, NORMAS)", dCoex),
        ("expl.      dD (sube densidad)", dD),
        ("expl.      dM (baja modularidad)", dM),
        ("expl.      gap_none (1-|corr grado-norma| de partida)", gap_none),
    ]
    for name, xs in tests:
        r, n = pearson(xs, g_ewc)
        p = approx_p(r, n)
        flag = ""
        if p == p and p < thr:
            flag = "  <-- ROBUSTO"
        elif p == p and p < 0.05:
            flag = "  (nominal, NO pasa Bonferroni)"
        print(f"  {name:<48} r={r:+.3f}  p~{p:.3f}{flag}")

    print("\n" + "=" * 66)
    print("VEREDICTO SEGUN EL PRERREGISTRO")
    print("=" * 66)
    rC, nC = pearson(dC, g_ewc);      pC = approx_p(rC, nC)
    rX, nX = pearson(dCoex, g_ewc);   pX = approx_p(rX, nX)
    topo = (pC == pC and pC < thr and rC > 0)
    norm = (pX == pX and pX < thr and rX > 0)

    # ¿estan dC y dCoex acopladas entre si? (la formula las liga)
    r_couple, _ = pearson(dC, dCoex)

    if topo and not norm:
        print("  >>> HIPOTESIS TOPOLOGICA SOPORTADA: el gain sube con dC y no")
        print("      se explica por dCoex. Omega gana donde hace topologia.")
    elif norm and not topo:
        print("  >>> HIPOTESIS DE NORMAS SOPORTADA: el gain sube con dCoex.")
        print("      El mecanismo es esencialmente igualar normas de fila.")
    elif topo and norm:
        print("  >>> AMBAS correlacionan con el gain. Hay que distinguir DOS casos:")
        print(f"      Correlacion dC<->dCoex entre si: r={r_couple:+.3f}")
        if abs(r_couple) > 0.6:
            print("      |r| alto -> MIXTO POR ACOPLAMIENTO: C y Coex se mueven")
            print("      juntas (lados opuestos de la misma fraccion), NO se pueden")
            print("      separar con estos datos. Esto NO evidencia mecanismo doble;")
            print("      solo dice que el experimento no las desacopla. Hace falta")
            print("      un brazo que suba C sin bajar Coex (o al reves).")
        else:
            print("      |r| bajo -> C y Coex se mueven con cierta INDEPENDENCIA.")
            print("      Mirar la tabla por semilla: si en unas domina dC y en otras")
            print("      dCoex, eso APUNTA (no prueba) a regimenes distintos segun")
            print("      la red. Es la senal mas cercana al mecanismo doble que")
            print("      estos datos permiten, pero necesita mas semillas y un")
            print("      diseno que desacople para confirmarse.")
    else:
        print("  >>> NINGUNA correlacion robusta. El patron bimodal de la tabla")
        print("      NO se explica por la estructura medida: sobre estos datos")
        print("      es indistinguible de RUIDO. La hipotesis del mecanismo")
        print("      dependiente de red NO se sostiene aqui. Se reporta negativo.")

    # Test dedicado del mecanismo doble: ¿que componente domina en cada semilla?
    print("\n" + "=" * 66)
    print("SENAL DE MECANISMO DOBLE (exploratoria, NO concluyente con n=10)")
    print("=" * 66)
    print("Para cada semilla, cual movimiento normalizado es mayor: dC o dCoex.")
    print("Si las semillas donde gana omega se reparten entre 'domina-C' y")
    print("'domina-Coex', eso sugiere dos regimenes. Si todas son del mismo")
    print("tipo, hay un solo mecanismo.")
    # normalizar cada delta por su desviacion para poder compararlos
    sdC = st.stdev(dC) if len(dC) > 1 and st.stdev(dC) > 0 else 1.0
    sdX = st.stdev(dCoex) if len(dCoex) > 1 and st.stdev(dCoex) > 0 else 1.0
    print(f"\n{'seed':<7}{'gain_ewc':>9}{'dC_norm':>9}{'dCoex_norm':>11}{'domina':>10}")
    domina_C = domina_X = 0
    for i, s in enumerate(seeds):
        zc = dC[i] / sdC; zx = dCoex[i] / sdX
        dom = "C" if zc > zx else "Coex"
        if g_ewc[i] > 0:
            if zc > zx: domina_C += 1
            else: domina_X += 1
        print(f"{s:<7}{g_ewc[i]:>+9.3f}{zc:>+9.2f}{zx:>+11.2f}{dom:>10}")
    print(f"\nEntre las semillas donde omega GANA: domina-C en {domina_C}, "
          f"domina-Coex en {domina_X}.")
    if domina_C > 0 and domina_X > 0:
        print("  -> Reparto MIXTO: senal (debil, n pequeno) compatible con dos")
        print("     regimenes. Justifica una corrida mayor con diseno desacoplado.")
    else:
        print("  -> Un solo tipo domina: sin senal de mecanismo doble aqui.")

    print("\nTabla por semilla (para inspeccion, NO para redisenar el test):")
    print(f"{'seed':<7}{'gain_ewc':>9}{'dC':>9}{'dCoex':>10}{'corr_dn':>9}")
    for i, s in enumerate(seeds):
        print(f"{s:<7}{g_ewc[i]:>+9.3f}{dC[i]:>+9.3f}{dCoex[i]:>+10.3f}"
              f"{corr_dn_none[i]:>9.3f}")


if __name__ == "__main__":
    main()
