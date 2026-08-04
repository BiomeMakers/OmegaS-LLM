#!/usr/bin/env python3
"""
Barrido y evaluacion del brazo rownorm con SU PROPIA rejilla.

POR QUE
    En la corrida original (run_overnight.py) rownorm fue el UNICO brazo que
    no se barrio: build_tuning_jobs() solo barre wd, ewc, omega_raw y
    omega_lib. En la evaluacion, rownorm se lanzo con
    OMEGA_TARGET = optimo de omega_lib (0.03), un valor calibrado para otra
    penalizacion, y colapso (retencion 0.207, HumanEval 0.027, y ya 0.153
    tras la PRIMERA tarea, o sea el modelo quedo danado, no olvidadizo).
    Eso deja sin responder la pregunta que decide si el mecanismo es
    novedoso: es Omega-S separable de igualar normas de fila, o no.

QUE HACE
    Fase 1: barre rownorm sobre RN_GRID en las semillas de ajuste (42, 123).
    Fase 2: fija el optimo y evalua rownorm en las 10 semillas de evaluacion.
    Fase 3: compara contra omega_lib leyendo results/merged_10seeds.json.

LA REJILLA BAJA DE 0.03 A PROPOSITO
    OMG_GRID original = [0.03, 0.1, 0.3, 0.5]. rownorm ya colapso EN 0.03,
    que es su valor mas bajo. Barrer esa rejilla daria cuatro celdas rotas y
    cero informacion. RN_GRID baja dos ordenes de magnitud y conserva el 0.03
    como ancla, para verificar que se reproduce el colapso conocido.

SEGURIDAD
    NO toca results/ ni results_tuning/. Escribe en directorios propios.
    Reanudable: salta las celdas cuyo json ya existe y es valido.

USO
    cd /workspace/omega-s
    python run_rownorm.py --smoke            # PRIMERO. Obligatorio. ~10 min.
    nohup python run_rownorm.py > rownorm.log 2>&1 &
    tail -f rownorm.log
"""
import os, sys, json, glob, time, subprocess, argparse, statistics as st

ROOT = "/workspace/omega-s"
PY   = sys.executable

TUNE_DIR = os.path.join(ROOT, "results_rownorm_tune")
EVAL_DIR = os.path.join(ROOT, "results_rownorm_eval")
REF_JSON = os.path.join(ROOT, "results", "merged_10seeds.json")

TUNE_SEEDS = [42, 123]
EVAL_SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
RN_GRID    = [0.001, 0.003, 0.01, 0.03]
BEST_WD    = "0.05"          # el mismo que uso la corrida original


# ---------------------------------------------------------------- utilidades
def ngpu():
    try:
        o = subprocess.run(["nvidia-smi", "--list-gpus"],
                           capture_output=True, text=True, timeout=20)
        return max(1, len(o.stdout.strip().splitlines()))
    except Exception:
        return 1


def free_gpus():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        if o.returncode != 0:
            return [0]
        return [int(l.split(",")[0]) for l in o.stdout.strip().splitlines()
                if int(l.split(",")[1]) < 1024]
    except Exception:
        return [0]


def cell_done(path):
    """Una celda esta hecha si su json existe, parsea y trae una fila."""
    try:
        rows = json.load(open(path))
        return isinstance(rows, list) and len(rows) >= 1 \
               and "retention_pct" in rows[0]
    except Exception:
        return False


def launch(env_extra, args, log, out):
    e = dict(os.environ)
    e["PYTHONPATH"] = ROOT
    e.update(env_extra)
    cmd = [PY, "experiments/rerun_retention.py"] + args + ["--out", out]
    return subprocess.Popen(cmd, cwd=ROOT, env=e,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)


def run_pool(jobs, phase):
    if not jobs:
        print("[" + phase + "] nada que hacer, todo estaba ya calculado",
              flush=True)
        return
    pending, running = list(jobs), {}
    print("[" + phase + "] " + str(len(pending)) + " trabajos, " +
          str(ngpu()) + " GPUs", flush=True)
    while pending or running:
        for gpu, (proc, tag) in list(running.items()):
            if proc.poll() is not None:
                print("  [" + phase + "] fin " + tag +
                      " (GPU " + str(gpu) + ")", flush=True)
                del running[gpu]
        libres = [g for g in free_gpus() if g not in running]
        for gpu in libres:
            if not pending:
                break
            job = pending.pop(0)
            env = dict(job["env"])
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            outp = os.path.join(job["dir"], job["tag"] + ".json")
            logp = os.path.join(job["dir"], job["tag"] + ".log")
            proc = launch(env, job["args"], logp, outp)
            running[gpu] = (proc, job["tag"])
            print("  [" + phase + "] lanzo " + job["tag"] +
                  " en GPU " + str(gpu), flush=True)
            time.sleep(15)
        time.sleep(30)
    print("[" + phase + "] COMPLETO", flush=True)


# ------------------------------------------------------------------- fase 1
def build_tuning_jobs(smoke):
    seeds = TUNE_SEEDS[:1] if smoke else TUNE_SEEDS
    grid  = RN_GRID[:1] if smoke else RN_GRID
    jobs = []
    for seed in seeds:
        for t in grid:
            tag = "rownorm_" + str(t) + "_s" + str(seed)
            if cell_done(os.path.join(TUNE_DIR, tag + ".json")):
                print("  (ya hecha) " + tag, flush=True)
                continue
            args = ["--cell", "--seed", str(seed), "--arm", "rownorm",
                    "--best-wd", BEST_WD]
            if smoke:
                args.append("--smoke")
            jobs.append(dict(env={"OMEGA_TARGET": str(t)}, dir=TUNE_DIR,
                             tag=tag, args=args))
    return jobs


def pick_best():
    """Media de retencion por valor de target. Devuelve (mejor, tabla).

    Escribe la tabla como JSON plano. La version original guardaba tuplas
    como clave y pick_best petaba al serializar; por eso optimos_elegidos
    salio {} en su dia. Aqui las claves son cadenas.
    """
    rows = []
    for f in glob.glob(os.path.join(TUNE_DIR, "*.json")):
        try:
            rows += json.load(open(f))
        except Exception:
            pass
    agg = {}
    for r in rows:
        if r.get("arm") != "rownorm":
            continue
        t = r.get("omega_target")
        if t is None:
            continue
        agg.setdefault(str(t), []).append(r["retention_pct"])

    tabla = {}
    for k, v in agg.items():
        tabla[k] = dict(n=len(v), media=st.mean(v),
                        desv=(st.stdev(v) if len(v) > 1 else 0.0))
    if not tabla:
        return None, tabla
    mejor = max(tabla.items(), key=lambda kv: kv[1]["media"])[0]
    json.dump(dict(tabla=tabla, mejor=mejor),
              open(os.path.join(TUNE_DIR, "rownorm_optimo.json"), "w"),
              indent=2)
    return float(mejor), tabla


# ------------------------------------------------------------------- fase 2
def build_eval_jobs(target, smoke):
    seeds = EVAL_SEEDS[:1] if smoke else EVAL_SEEDS
    jobs = []
    for seed in seeds:
        tag = "rownorm_s" + str(seed)
        if cell_done(os.path.join(EVAL_DIR, tag + ".json")):
            print("  (ya hecha) " + tag, flush=True)
            continue
        args = ["--cell", "--seed", str(seed), "--arm", "rownorm",
                "--best-wd", BEST_WD]
        if smoke:
            args.append("--smoke")
        jobs.append(dict(env={"OMEGA_TARGET": str(target)}, dir=EVAL_DIR,
                         tag=tag, args=args))
    return jobs


# ------------------------------------------------------------------- fase 3
def comparar():
    rows = []
    for f in glob.glob(os.path.join(EVAL_DIR, "*.json")):
        try:
            rows += json.load(open(f))
        except Exception:
            pass
    rn = {r["seed"]: r for r in rows if r.get("arm") == "rownorm"}
    if not os.path.exists(REF_JSON):
        print("\nNo encuentro " + REF_JSON + ", no puedo comparar.")
        print("Resultados de rownorm guardados en " + EVAL_DIR)
        return
    ref = json.load(open(REF_JSON))
    lib = {r["seed"]: r for r in ref if r["arm"] == "omega_lib"}
    non = {r["seed"]: r for r in ref if r["arm"] == "none"}

    comunes = sorted(set(rn) & set(lib))
    if not comunes:
        print("\nSin semillas en comun todavia.")
        return

    print("\n" + "=" * 74)
    print("COMPARACION: rownorm ajustado  vs  omega_lib  vs  none")
    print("=" * 74)
    print("            capacidad absoluta (HumanEval tras prosa)")
    print("semilla   rownorm   omega_lib      none   |  ret rownorm  ret omega")
    gana_lib = 0
    for s in comunes:
        a = rn[s]["humaneval_after_B"]
        b = lib[s]["humaneval_after_B"]
        c = non[s]["humaneval_after_B"]
        gana_lib += b > a
        print("%-9s %8.3f %10.3f %9.3f   | %10.3f %10.3f"
              % (s, a, b, c, rn[s]["retention_pct"], lib[s]["retention_pct"]))
    n = len(comunes)
    print("-" * 74)
    print("media     %8.3f %10.3f %9.3f"
          % (st.mean(rn[s]["humaneval_after_B"] for s in comunes),
             st.mean(lib[s]["humaneval_after_B"] for s in comunes),
             st.mean(non[s]["humaneval_after_B"] for s in comunes)))
    print("\nomega_lib gana a rownorm en " + str(gana_lib) + "/" + str(n) +
          " semillas (capacidad absoluta)")
    if n >= 8:
        from math import comb
        p = sum(comb(n, k) for k in range(gana_lib, n + 1)) / 2 ** n
        print("test de signos, una cola: p = %.4f" % p)
    print("""
COMO LEERLO
  omega_lib gana claro   -> hay algo que igualar normas de fila no captura.
                            La afirmacion vuelve al abstract, con evidencia.
  empatan                -> el mecanismo ES igualar normas de fila. Sigue
                            siendo util y publicable, pero es otro relato.
  rownorm gana           -> el termino de Tr(A^3) estorba; el camino es el
                            termino simple. Tambien informativo.
""")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 celdas con tiny-gpt2 para verificar el pipeline")
    ap.add_argument("--solo-comparar", action="store_true",
                    help="no lanza nada, solo imprime la comparacion")
    a = ap.parse_args()

    os.makedirs(TUNE_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    if a.solo_comparar:
        comparar()
        return

    print("REJILLA rownorm: " + str(RN_GRID))
    print("semillas de ajuste: " + str(TUNE_SEEDS))
    print("semillas de evaluacion: " + str(EVAL_SEEDS))
    print("modo: " + ("SMOKE" if a.smoke else "REAL") + "\n", flush=True)

    run_pool(build_tuning_jobs(a.smoke), "ajuste")

    mejor, tabla = pick_best()
    print("\n" + "-" * 50)
    print("BARRIDO DE rownorm")
    print("%-10s %8s %10s %6s" % ("target", "media", "desv", "n"))
    for k in sorted(tabla, key=float):
        v = tabla[k]
        print("%-10s %8.4f %10.4f %6d" % (k, v["media"], v["desv"], v["n"]))
    if mejor is None:
        print("\nNo hay resultados de ajuste. Revisa los .log de " + TUNE_DIR)
        return
    print("\n>>> optimo de rownorm: " + str(mejor))
    print("-" * 50 + "\n", flush=True)

    if a.smoke:
        print("SMOKE OK. El pipeline escribe json y el barrido agrega bien.")
        print("Ahora borra los directorios de smoke y lanza la corrida real:")
        print("  rm -rf " + TUNE_DIR + " " + EVAL_DIR)
        print("  nohup python run_rownorm.py > rownorm.log 2>&1 &")
        return

    run_pool(build_eval_jobs(mejor, a.smoke), "eval")
    comparar()
    print("\nAPAGA EL POD.")


if __name__ == "__main__":
    main()
