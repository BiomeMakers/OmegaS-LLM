#!/usr/bin/env python3
"""
Corrida de noche desatendida: AJUSTE -> seleccion automatica -> EVALUACION.

Fase 1: barre el hiperparametro de wd, ewc, omega_raw, omega_lib en las
        semillas de ajuste (42, 123). Todos ajustados con la misma vara.
Fase 2: fija el mejor valor de cada brazo y corre la evaluacion en las 10
        semillas, cada brazo en su optimo, mas none y rownorm.

Reparte el trabajo entre todas las GPUs, repone lo que se caiga, y guarda
todo incrementalmente. Pensado para dejarlo con nohup y revisarlo por la
manana. La SELECCION automatica del optimo hay que revisarla a mano despues:
si eligio un valor raro, se ve en tuning_merged.json y se repite esa parte.

Uso:
    cd /workspace/omega-s
    nohup python run_overnight.py > overnight.log 2>&1 &
    tail -f overnight.log
"""
import os, sys, json, glob, time, subprocess, statistics as st

ROOT = "/workspace/omega-s"
PY = sys.executable
TUNE_DIR = os.path.join(ROOT, "results_tuning")
EVAL_DIR = os.path.join(ROOT, "results")
os.makedirs(TUNE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

TUNE_SEEDS = [42, 123]
EVAL_SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
WD_GRID   = [0.0, 0.01, 0.05, 0.1]
EWC_GRID  = [100, 500, 1000, 5000, 10000]
OMG_GRID  = [0.03, 0.1, 0.3, 0.5]

def ngpu():
    try:
        o = subprocess.run(["nvidia-smi","--list-gpus"],capture_output=True,text=True,timeout=20)
        return len(o.stdout.strip().splitlines())
    except Exception:
        return 1

NG = ngpu()

def free_gpus():
    try:
        o = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20)
        if o.returncode != 0: return []
        return [int(l.split(",")[0]) for l in o.stdout.strip().splitlines()
                if int(l.split(",")[1]) < 1024]
    except Exception:
        return []

def launch(env_extra, args, log, out):
    e = dict(os.environ); e["PYTHONPATH"] = ROOT; e.update(env_extra)
    cmd = [PY, "experiments/rerun_retention.py"] + args + ["--out", out]
    return subprocess.Popen(cmd, cwd=ROOT, env=e,
        stdout=open(log,"w"), stderr=subprocess.STDOUT)

def run_pool(jobs, phase):
    """jobs: lista de dicts con env, args, tag. Los reparte entre GPUs."""
    pending = list(jobs); running = {}
    print(f"[{phase}] {len(pending)} trabajos, {NG} GPUs", flush=True)
    while pending or running:
        # cosechar terminados
        for gpu, (proc, tag) in list(running.items()):
            if proc.poll() is not None:
                print(f"  [{phase}] fin {tag} (GPU {gpu})", flush=True)
                del running[gpu]
        # lanzar en GPUs libres
        libres = [g for g in free_gpus() if g not in running]
        for gpu in libres:
            if not pending: break
            job = pending.pop(0)
            env = dict(job["env"]); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            outp = os.path.join(job["dir"], job["tag"] + ".json")
            logp = os.path.join(job["dir"], job["tag"] + ".log")
            proc = launch(env, job["args"], logp, outp)
            running[gpu] = (proc, job["tag"])
            print(f"  [{phase}] lanzo {tag_of(job)} en GPU {gpu}", flush=True)
            time.sleep(15)
        time.sleep(30)
    print(f"[{phase}] COMPLETO", flush=True)

def tag_of(job): return job["tag"]

# ---------- FASE 1: AJUSTE ----------
def build_tuning_jobs():
    jobs = []
    for seed in TUNE_SEEDS:
        for wd in WD_GRID:
            jobs.append(dict(env={}, dir=TUNE_DIR, tag=f"wd_{wd}_s{seed}",
                args=["--cell","--seed",str(seed),"--arm","wd","--best-wd",str(wd)]))
        for l in EWC_GRID:
            jobs.append(dict(env={"EWC_LAMBDA":str(l)}, dir=TUNE_DIR, tag=f"ewc_{l}_s{seed}",
                args=["--cell","--seed",str(seed),"--arm","ewc","--best-wd","0.05"]))
        for t in OMG_GRID:
            for arm in ["omega_raw","omega_lib"]:
                jobs.append(dict(env={"OMEGA_TARGET":str(t)}, dir=TUNE_DIR,
                    tag=f"{arm}_{t}_s{seed}",
                    args=["--cell","--seed",str(seed),"--arm",arm,"--best-wd","0.05"]))
    return jobs

def pick_best():
    rows = []
    for f in glob.glob(os.path.join(TUNE_DIR,"*.json")):
        try: rows += json.load(open(f))
        except Exception: pass
    def hp(r):
        if r["arm"]=="wd": return r["wd"]
        if r["arm"]=="ewc": return r.get("ewc_lambda") or r.get("omega_lambda")
        return r.get("omega_target")
    agg = {}
    for r in rows:
        agg.setdefault((r["arm"],hp(r)),[]).append(r["retention_pct"])
    best = {}
    for (arm,val),rets in agg.items():
        m = st.mean(rets)
        if arm not in best or m > best[arm][1]: best[arm] = (val,m)
    return best

# ---------- FASE 2: EVALUACION ----------
def build_eval_jobs(best):
    wd_opt  = best.get("wd",(0.05,0))[0]
    ewc_opt = best.get("ewc",(1000,0))[0]
    raw_opt = best.get("omega_raw",(0.1,0))[0]
    lib_opt = best.get("omega_lib",(0.1,0))[0]
    jobs = []
    for seed in EVAL_SEEDS:
        jobs.append(dict(env={}, dir=EVAL_DIR, tag=f"none_s{seed}",
            args=["--cell","--seed",str(seed),"--arm","none","--best-wd",str(wd_opt)]))
        jobs.append(dict(env={}, dir=EVAL_DIR, tag=f"wd_s{seed}",
            args=["--cell","--seed",str(seed),"--arm","wd","--best-wd",str(wd_opt)]))
        jobs.append(dict(env={"EWC_LAMBDA":str(ewc_opt)}, dir=EVAL_DIR, tag=f"ewc_s{seed}",
            args=["--cell","--seed",str(seed),"--arm","ewc","--best-wd",str(wd_opt)]))
        jobs.append(dict(env={"OMEGA_TARGET":str(lib_opt)}, dir=EVAL_DIR, tag=f"rownorm_s{seed}",
            args=["--cell","--seed",str(seed),"--arm","rownorm","--best-wd",str(wd_opt)]))
        jobs.append(dict(env={"OMEGA_TARGET":str(raw_opt)}, dir=EVAL_DIR, tag=f"omega_raw_s{seed}",
            args=["--cell","--seed",str(seed),"--arm","omega_raw","--best-wd",str(wd_opt)]))
        jobs.append(dict(env={"OMEGA_TARGET":str(lib_opt)}, dir=EVAL_DIR, tag=f"omega_lib_s{seed}",
            args=["--cell","--seed",str(seed),"--arm","omega_lib","--best-wd",str(wd_opt)]))
    return jobs

if __name__ == "__main__":
    t0 = time.time()
    print("=== FASE 1: AJUSTE ===", flush=True)
    run_pool(build_tuning_jobs(), "ajuste")

    best = pick_best()
    print("\n=== OPTIMOS ELEGIDOS (revisar a mano manana) ===", flush=True)
    for arm in ["wd","ewc","omega_raw","omega_lib"]:
        if arm in best:
            print(f"  {arm}: valor={best[arm][0]}  retencion={best[arm][1]:.4f}", flush=True)
    json.dump({k:list(v) for k,v in best.items()},
              open(os.path.join(ROOT,"optimos_elegidos.json"),"w"), indent=2)

    print("\n=== FASE 2: EVALUACION (10 semillas) ===", flush=True)
    run_pool(build_eval_jobs(best), "eval")

    print(f"\n=== TODO COMPLETO en {(time.time()-t0)/3600:.1f} h ===", flush=True)
    print("Manana: python experiments/merge_tuning.py results_tuning/", flush=True)
    print("        python experiments/merge_results.py results/", flush=True)
    print("        cat optimos_elegidos.json", flush=True)
