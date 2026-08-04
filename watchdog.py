import os, sys, json, glob, time, subprocess
ROOT="/workspace/omega-s"; OUT=os.path.join(ROOT,"results")
SEEDS=[42,123,456,789,1011,2022,3033,4044,5055,6066]
ARMS=["none","wd","ewc","rownorm","omega_raw","omega_lib"]
CELLS=[(s,a) for s in SEEDS for a in ARMS]
BEST_WD=sys.argv[1] if len(sys.argv)>1 else "0.05"

def done():
    got=set()
    for f in glob.glob(os.path.join(OUT,"*.json")):
        try:
            for r in json.load(open(f)): got.add((r["seed"],r["arm"]))
        except Exception: pass
    return got

def free():
    try:
        o=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20)
        if o.returncode!=0: return []
        return [int(l.split(",")[0]) for l in o.stdout.strip().splitlines()
                if int(l.split(",")[1]) < 1024]
    except Exception:
        print("  (nvidia-smi no responde, reintento)",flush=True); return []

def launch(g,i):
    sd,ar=CELLS[i]; tag=f"c{i}_g{g}"
    e=dict(os.environ); e["CUDA_VISIBLE_DEVICES"]=str(g); e["PYTHONPATH"]=ROOT
    cmd=[sys.executable,"experiments/rerun_retention.py","--cells",str(i),
         "--best-wd",BEST_WD,"--out",os.path.join(OUT,f"{tag}.json")]
    subprocess.Popen(cmd,cwd=ROOT,env=e,
        stdout=open(os.path.join(OUT,f"{tag}.log"),"w"),stderr=subprocess.STDOUT)
    print(f"  -> celda {i} (semilla {sd}, {ar}) en GPU {g}",flush=True)

os.makedirs(OUT,exist_ok=True); lanz=set()
print(f"Vigilante v3. {len(CELLS)} celdas, wd={BEST_WD}.\n",flush=True)
while True:
    h=done(); pend=[i for i,c in enumerate(CELLS) if c not in h and i not in lanz]
    lib=free()
    print(f"[{time.strftime('%H:%M:%S')}] hechas {len(h)}/{len(CELLS)} | "
          f"pendientes {len(pend)} | GPUs libres {lib}",flush=True)
    if len(h)>=len(CELLS):
        print("\nTODAS TERMINADAS. python experiments/merge_results.py results/",flush=True)
        break
    for g in lib:
        if not pend: break
        i=pend.pop(0); launch(g,i); lanz.add(i); time.sleep(20)
    lanz={i for i in lanz if CELLS[i] not in h}
    time.sleep(60)
