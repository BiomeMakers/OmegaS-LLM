# =============================================================================
# Omega-S : Re-corrida de retención (RunPod)
# Sustituye a fase2_omega.py y fase3_semilla*.py
# =============================================================================
# Por qué existe: en los scripts originales la penalización se calculaba con
# `hutchinson_tr_a3(p.data)`. `.data` desacopla del grafo de autograd, así que
# el término se sumaba a la pérdida como una CONSTANTE y su gradiente era CERO.
# El regularizador nunca tocó los pesos. Ver AUDIT.md.
#
# Qué cambia:
#   1. Penalización CONECTADA, sobre el peso efectivo W_base + s*(B@A).
#   2. Verificación de gradiente ANTES de entrenar. Si es cero, aborta.
#   3. RNG DEDICADO para la penalización. Antes consumía el RNG global y
#      desincronizaba el brazo Omega respecto al baseline: era otra semilla,
#      no otro método. Ese era el confound que producía las diferencias.
#   4. Semillas de verdad distintas: 42, 123, 456. Antes fase3_semilla3.py era
#      copia de fase3_semilla2.py y ambos tenían SEED=123.
#   5. CINCO brazos, incluidos los baselines que faltaban:
#        none      sin regularizar
#        wd        weight decay AJUSTADO por rejilla, no un valor por defecto
#        ewc       Elastic Weight Consolidation, baseline canónico desde 2017
#        omega_raw Tr((WW^T)^3) crudo, el de los experimentos originales
#        omega_lib StochasticOmegaS del paquete, el que describen el preprint
#                  y la patente. Correr ambos contesta de paso si librería y
#                  experimentos deben unificarse.
#   6. HumanEval pass@1, la misma métrica del número publicado (83.03/81.07).
#   7. OMEGA_LAMBDA se CALIBRA. El 0.05 anterior se ajustó contra una constante.
#
# Uso:
#   python rerun_retention.py --smoke                            # ~10 min
#   python rerun_retention.py --cell --seed 42 --arm omega_raw   # UNA celda
#   python rerun_retention.py --all                              # rejilla
# =============================================================================

import os, sys, json, time, math, argparse, statistics, itertools, subprocess, tempfile
import torch, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset

# El paquete omega_s vive en la RAIZ del repo, pero al lanzar
# `python experiments/rerun_retention.py` Python pone experiments/ en sys.path
# y no la raiz, asi que el brazo omega_lib moria con ModuleNotFoundError. Solo
# lo sufria omega_lib, porque es el unico que importa el paquete.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DECISION_RULE = """
REGLA DE DECISION, PRERREGISTRADA (escrita antes de mirar ningun resultado)

Omega-S se considera VALIDADO si y solo si:
  (a) supera a weight decay AJUSTADO en retencion en >= 2 de 3 semillas,
  (b) sin pagar mas de 1pp adicional de plasticidad frente a ese mismo brazo,
  (c) y queda dentro de una desviacion tipica de EWC o por encima.
Si no se cumple, el claim de retencion NO se publica y el preprint de LLM se
reescribe sin el. Vale por separado para omega_raw y para omega_lib.
"""

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS    = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
# Ajustar sobre una semilla que esta DENTRO del conjunto de evaluacion es lo
# que produjo la regresion al ampliar de 2 a 10 y costo un parrafo de
# limitaciones. TUNE_SEED es disjunta y el assert impide que vuelva a pasar.
TUNE_SEED = int(os.environ.get("TUNE_SEED", "7"))
assert TUNE_SEED not in SEEDS, "TUNE_SEED debe ser disjunta de SEEDS"
ARMS     = ["none", "wd", "ewc", "rownorm", "omega_raw", "omega_lib", "cos_full", "cos_noCoex", "cos_composite",
            "omega_spectral",  # CONTROL espectral (sigma_max^6), NO una variante de Omega
            "l2sp",            # rival DATA-FREE que faltaba (Li et al. 2018)
            "omega_minv"]      # M = 1/lambda_2, LA ORIENTACION PUBLICADA (ret 0.8408)
WD_GRID  = [0.0, 0.01, 0.05, 0.1]

MODEL_ID     = os.environ.get("MODEL_ID", "NousResearch/Meta-Llama-3-8B")
# configurable para comprobar si un resultado transfiere a otro modelo.
# OJO: los valores absolutos entre modelos distintos NO son comparables;
# solo valen las comparaciones dentro del mismo modelo.
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1
LORA_TARGETS = ["q_proj", "v_proj"]  # 1b/fase2: pareado con las 10 semillas
LR, BATCH_SIZE, GRAD_ACCUM = 2e-4, 2, 4
MAX_SEQ_LEN, MAX_SAMPLES   = 512, 5000
EPOCHS_A = EPOCHS_B = 1

HUMANEVAL_N  = 164
MAX_NEW_TOK  = 256
GEN_TEMP     = 0.1
HE_BATCH     = 32      # generacion por lotes: 8-10x mas rapido que de uno en uno

OMEGA_EVERY_K = 10
OMEGA_PROBES  = 16     # 3 daba un estimador dominado por ruido de muestreo
OMEGA_MODULES = 8      # muestreo de modulos; W W^T cuesta O(N^2 * in)
EWC_LAMBDA    = float(os.environ.get("EWC_LAMBDA", "1000.0"))  # barrible
L2SP_LAMBDA   = float(os.environ.get("L2SP_LAMBDA", "1.0"))    # barrible
# --- Fase 1a: construccion coseno y diagnostico ---
COS_SIGN     = float(os.environ.get("COS_SIGN", "1.0"))
# Ablacion por modulo: "" = todos, "q_proj" o "v_proj" = solo ese tipo.
# El filtro actua sobre la PENALIZACION, no sobre LORA_TARGETS, para que el
# adaptador sea identico en los tres brazos.
OMEGA_ONLY   = os.environ.get("OMEGA_ONLY", "").strip()
# Colocacion explicita del adaptador: lista de nombres COMPLETOS de modulo,
# separados por coma. Si esta vacia se usa LORA_TARGETS, que es la convencion.
LORA_MODULES = [m.strip() for m in
                os.environ.get("LORA_MODULES", "").split(",") if m.strip()]
DIAG_EVERY   = int(os.environ.get("DIAG_EVERY", "25"))
DIAG_K       = 4
PHASE1A_STEPS = int(os.environ.get("PHASE1A_STEPS", "200"))


# ===========================================================================
# PENALIZACIONES
# ===========================================================================
def iter_effective_weights(model):
    """W_eff = W_base + scaling * (B @ A). Conectado al grafo."""
    for name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        base = getattr(mod, "base_layer", None)
        if base is None or not hasattr(base, "weight"):
            continue
        for key in mod.lora_A.keys():
            A = mod.lora_A[key].weight
            B = mod.lora_B[key].weight
            s = mod.scaling[key] if isinstance(mod.scaling, dict) else mod.scaling
            delta = s * (B @ A)
            W = base.weight
            if W.shape != delta.shape and tuple(W.shape) == tuple(delta.shape)[::-1]:
                delta = delta.t()
            yield name, W + delta
            break


SPECTRAL_ITERS = int(os.environ.get("SPECTRAL_ITERS", "400"))
# 400 y no 20: MEDIDO sobre seis matrices aleatorias de 64x64 a 2048x1024,
# el error relativo mediano de sigma_max^6 pasa de 5.2e-2 (20 iteraciones)
# a 2.0e-6 (400), y el PEOR caso de 1.4e-1 a 3.9e-6. Con 20 el control
# llegaria a la comparacion con un 14% de error en el peor modulo, y la
# calibracion de lambda heredaria ese ruido. El coste son ~2.2 s por
# modulo en CPU para 4096x4096, o sea el mismo orden que las 16 sondas de
# Hutchinson, y en GPU es despreciable.


def sigma_max_pow6(W, gen, n_iter=None):
    """sigma_max(W)^6 = lambda_max(W W^T)^3. CONTROL, NO ES OMEGA.

    Sin grafo, sin triangulos, sin modularidad: solo la direccion dominante.
    Si este brazo retiene como Omega, entonces Omega es encogimiento de la
    direccion dominante con nombre nuevo, y eso hay que saberlo.

    Iteracion de potencia sobre W W^T sin gradiente para hallar el vector
    propio, y luego UN cociente de Rayleigh diferenciable con ese vector ya
    convergido. Asi el gradiente que llega a W es el de lambda_max y no el
    del bucle entero, que ademas seria inestable.
    """
    Wf = W.float()
    n_iter = SPECTRAL_ITERS if n_iter is None else n_iter
    v = torch.randn(Wf.shape[0], generator=gen).to(Wf.device)
    v = v / (v.norm() + 1e-12)
    with torch.no_grad():
        for _ in range(n_iter):
            v = Wf @ (Wf.t() @ v)
            nv = v.norm()
            if nv < 1e-30:
                break
            v = v / nv
    lam = v @ (Wf @ (Wf.t() @ v))      # lambda_max(W W^T), diferenciable
    return lam ** 3                    # = sigma_max(W)^6


def hutchinson_tr_a3(W, n, gen):
    """Tr((W W^T)^3). El metodo de los experimentos originales."""
    total, Wf = 0.0, W.float()
    for _ in range(n):
        v = (torch.randint(0, 2, (Wf.shape[0],), generator=gen,
                           dtype=torch.float32) * 2 - 1).to(Wf.device)
        z = Wf @ (Wf.t() @ v)
        z = Wf @ (Wf.t() @ z)
        z = Wf @ (Wf.t() @ z)
        total = total + (v @ z)
    return total / n



# ===========================================================================
# FASE 1a: construccion COSENO
# ===========================================================================
def cosine_A(W, zero_diag=True, eps=1e-8):
    """A = |cos(w_i,w_j)| entre filas de W, diagonal a cero.
    CALCADO a c_cosine de sat_test_v4.py: normaliza filas, |producto|, clip a
    [0,1], diagonal 0. Verificado en Mac: C/D 1.05 a 1.84 segun modulo.
    Invariante a la norma de fila por construccion (normalizar w_i no cambia
    ningun coseno): es lo que quita la dependencia mecanica del canal de grados.
    """
    Wf = W.float()
    Wn = Wf / (Wf.norm(dim=1, keepdim=True) + eps)
    A = (Wn @ Wn.t()).abs().clamp(0.0, 1.0)
    if zero_diag:
        A = A - torch.diag(torch.diagonal(A))
    return A


# ===========================================================================
# BRAZO COMPUESTO CON CONSTRUCCION COSENO
# Replica de StochasticOmegaS cambiando SOLO la construccion de A.
# ===========================================================================
OMEGA_EPS   = 1e-6
COS_C_SIGN  = float(os.environ.get("COS_C_SIGN", "1.0"))   # exponente sobre C


def cosine_composite(W, probes, gen):
    """log((M*Coex)/(C^s*D)) con A construida por coseno por filas.

    Las cuatro formulas son las de omega_s.py palabra por palabra; lo unico
    distinto es A. Devuelve un escalar conectado al grafo.
    """
    eps = OMEGA_EPS
    A = cosine_A(W, zero_diag=True)          # <- LA UNICA DIFERENCIA
    N = A.size(0)

    D = torch.mean(A)
    degrees = torch.sum(A, dim=1)
    Coex = torch.var(degrees) + eps

    # Hutchinson para Tr(A^3), normalizado por Frobenius al cubo (libreria)
    tr_A3 = 0.0
    for _ in range(probes):
        z = (torch.randint(0, 2, (N, 1), generator=gen,
                           dtype=torch.float32) * 2.0 - 1.0).to(A.device)
        tr_A3 = tr_A3 + torch.matmul(
            z.t(), torch.matmul(A, torch.matmul(A, torch.matmul(A, z)))).squeeze()
    tr_A3 = tr_A3 / probes
    C = tr_A3 / (torch.norm(A, p="fro") ** 3 + eps) + eps

    # Modularidad espectral por iteracion de potencia sobre el laplaciano
    v = torch.randn((N, 1), generator=gen, dtype=torch.float32).to(A.device)
    v = v - torch.mean(v)
    v = v / (torch.norm(v) + eps)
    max_deg = torch.max(degrees)
    for _ in range(3):
        Lv = (degrees.view(-1, 1) * v) - torch.matmul(A, v)
        v = ((2 * max_deg) * v - Lv)
        v = (v - torch.mean(v)) / (torch.norm(v) + eps)
    M_est = torch.abs(torch.matmul(
        v.t(), (degrees.view(-1, 1) * v) - torch.matmul(A, v)).squeeze()) + eps

    # C^s: s=+1 reproduce la libreria (minimizar SUBE C). s=-1 invierte solo
    # la direccion del canal de clustering, dejando M, Coex y D como estan.
    return torch.log((M_est * Coex) / ((C ** COS_C_SIGN) * D + eps))


def cosine_tr_a3_hutch(W, n, gen):
    """Tr(A^3) con A=coseno, estimador Hutchinson. Es lo que lleva el
    GRADIENTE del entrenamiento: insesgado y barato. A queda conectada al
    grafo porque W_eff lo esta."""
    A = cosine_A(W)
    total = 0.0
    for _ in range(n):
        v = (torch.randint(0, 2, (A.shape[0],), generator=gen,
                           dtype=torch.float32) * 2 - 1).to(A.device)
        z = A @ (A @ (A @ v))
        total = total + (v @ z)
    return total / n


def cosine_excess_CD_exact(W):
    """C/D EXACTO con A=coseno. SIN gradiente: es diagnostico, no perdida.
    Con 16 probes el ruido de Hutchinson aun tapa el cambio de C, asi que la
    trayectoria de dC se mide exacta. Mismo estadistico que la fase 1 y que
    verify_cosine.py del Mac."""
    with torch.no_grad():
        A = cosine_A(W).double()
        n = A.shape[0]
        A2 = A @ A
        C = (A * A2).sum() / (A2.sum() - torch.diagonal(A2).sum() + 1e-12)
        off = A.sum() - torch.diagonal(A).sum()
        D = off / (n * (n - 1) + 1e-12)
        return (C / (D + 1e-12)).item()


# [MERGING HOOK] Cuando 1a confirme que el canal sobrevive, la construccion de
# interferencia para merging reutiliza cosine_A cambiando el objeto: en vez de
# A = |cos| entre filas de un W_eff, se usa el coseno CRUZADO entre las filas
# de los dos deltas que se fusionan,
#     An = dW_A / ||fila||;  Bn = dW_B / ||fila||;  A = |An @ Bn.t()|
# que es la matriz de INTERFERENCIA del par y mide la tarea conjunta, no cada
# modelo por separado. Es el unico punto de cambio: el resto del circuito
# (Hutchinson para el gradiente, C/D exacto para el diagnostico, barrido de
# target y signo, puerta) se reutiliza tal cual.
def cosine_A_interference(dW_A, dW_B, eps=1e-8):
    An = dW_A.float(); Bn = dW_B.float()
    An = An / (An.norm(dim=1, keepdim=True) + eps)
    Bn = Bn / (Bn.norm(dim=1, keepdim=True) + eps)
    A = (An @ Bn.t()).abs().clamp(0.0, 1.0)
    if A.shape[0] == A.shape[1]:
        A = A - torch.diag(torch.diagonal(A))
    return A


_LIB_CORE = None
def assert_minv_disponible():
    """El paquete instalado tiene que traer el conmutador. Si no lo trae, el
    brazo correria la orientacion VIEJA en silencio y daria ~0.766 pareciendo
    creible. Esto es exactamente lo que pasa con el omega_s.py del repo
    publico a 7-ago, que no lleva las tres lineas."""
    import inspect
    try:
        from omega_s import StochasticOmegaS
    except ImportError:
        from omega_s.omega_s import StochasticOmegaS
    fuente = inspect.getsource(StochasticOmegaS)
    if "OMEGA_M_INV" not in fuente:
        raise RuntimeError(
            "El paquete omega_s instalado NO tiene el conmutador OMEGA_M_INV, "
            "asi que omega_minv correria la orientacion vieja en silencio. "
            "Anade a omega_s/omega_s.py, tras la linea de M_est:\n"
            '    if os.environ.get("OMEGA_M_INV", "0") == "1":\n'
            "        M_est = 1.0 / (M_est + self.epsilon)\n"
            "y el `import os` arriba del fichero.")
    if os.environ.get("OMEGA_M_INV") != "1":
        raise RuntimeError("OMEGA_M_INV deberia estar a 1 en este punto")
    print("  [omega_minv] conmutador presente y activo: M = 1/lambda_2")


def omega_lib_core():
    """StochasticOmegaS del paquete: log((M*Coex)/(C*D)) sobre sigmoid(|W W^T|)."""
    global _LIB_CORE
    if _LIB_CORE is None:
        try:
            from omega_s import StochasticOmegaS
        except ImportError:
            from omega_s.omega_s import StochasticOmegaS
        _LIB_CORE = StochasticOmegaS(num_samples=OMEGA_PROBES).to(DEVICE)
    return _LIB_CORE


def omega_pen(model, arm, gen):
    mods = list(iter_effective_weights(model))
    if OMEGA_ONLY:
        mods = [m for m in mods if m[0].endswith(OMEGA_ONLY)]
    if not mods:
        raise RuntimeError("Sin modulos LoRA. Revisa LORA_TARGETS / OMEGA_ONLY.")
    k = min(OMEGA_MODULES, len(mods))
    idx = torch.randperm(len(mods), generator=gen)[:k].tolist()
    if arm == "omega_raw":
        total = sum(hutchinson_tr_a3(mods[i][1], OMEGA_PROBES, gen) for i in idx)
    elif arm == "cos_composite":
        total = sum(cosine_composite(mods[i][1], OMEGA_PROBES, gen) for i in idx)
    elif arm in ("cos_full", "cos_noCoex"):
        # FASE 1a: SOLO el termino de clustering, +-log(Tr(A^3)_coseno).
        # Con el log-ratio compuesto el signo no se aisla; aqui si, y la
        # pregunta de 1a es exactamente si este canal esta vivo y mueve dC.
        # COS_SIGN=+1 sube C (direccion del objetivo actual y del FSRI),
        # COS_SIGN=-1 la baja (la prediccion mecanica opuesta).
        total = sum(COS_SIGN * torch.log(
            cosine_tr_a3_hutch(mods[i][1], OMEGA_PROBES, gen) + 1e-6)
            for i in idx)
    elif arm == "omega_spectral":
        # CONTROL espectral. Mismo muestreo de modulos y mismo RNG que el resto
        # de brazos para que la comparacion sea limpia.
        total = sum(sigma_max_pow6(mods[i][1], gen) for i in idx)
    elif arm in ("omega_lib", "omega_minv"):
        # Misma penalizacion; lo que cambia es la orientacion de M, que la
        # libreria lee de OMEGA_M_INV. run_cell la fija segun el brazo.
        core = omega_lib_core()
        total = sum(core(mods[i][1]) for i in idx)
    else:
        # ANTES esto era un `else` que caia en omega_lib_core(). Un brazo nuevo
        # sin rama, o con el nombre mal escrito, calculaba omega_lib EN SILENCIO
        # y devolvia diez celdas creibles y falsas. Mejor reventar en la celda 0.
        raise RuntimeError(
            "brazo desconocido en omega_pen: " + repr(arm) +
            ". Anade su rama explicita; no hay caida por defecto.")
    return total / k


PPL_SEQS = int(os.environ.get("PPL_SEQS", "64"))
PPL_LEN  = 512
_PPL_CACHE = {}


def _ppl_chunks(tok):
    """Trozos fijos de wikitext-2 test. Se cachean: son los MISMOS para
    todas las celdas, que es lo que hace comparables las perplejidades."""
    if "chunks" in _PPL_CACHE:
        return _PPL_CACHE["chunks"]
    ds = None
    for name in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(name, "wikitext-2-raw-v1", split="test")
            break
        except Exception as exc:
            print("  ppl: " + name + " no disponible (" + str(exc)[:80] + ")")
    if ds is None:
        raise RuntimeError("Sin corpus para perplejidad")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(PPL_SEQS, ids.shape[0] // PPL_LEN)
    chunks = [ids[i*PPL_LEN:(i+1)*PPL_LEN] for i in range(n)]
    print("  ppl: " + str(len(chunks)) + " secuencias de " + str(PPL_LEN))
    _PPL_CACHE["chunks"] = chunks
    return chunks


def perplexity(model, tok, tag=""):
    """Perplejidad en wikitext-2 test. GUARDIA DE CALIDAD: sin esto, un brazo
    que conserva el cociente de retencion mientras hunde el modelo pasa por
    ganador. Se mide en TODOS los brazos, no solo en los regularizados."""
    was_training = model.training
    model.eval()
    nll_total, tok_total = 0.0, 0
    with torch.no_grad():
        for ch in _ppl_chunks(tok):
            ids = ch.unsqueeze(0).to(DEVICE)
            loss = model(input_ids=ids, labels=ids).loss
            n_pred = ids.shape[1] - 1
            nll_total += float(loss) * n_pred
            tok_total += n_pred
    if was_training:
        model.train()
    ppl = math.exp(nll_total / max(tok_total, 1))
    print("  perplejidad " + tag + ": " + format(ppl, ".4f"))
    return ppl


def rownorm_pen(model, gen):
    """Control: penaliza SOLO la varianza de las normas de fila del peso
    efectivo. Cero topologia. Si Omega solo iguala normas, este brazo lo
    replica. Mismo muestreo de modulos y mismo RNG que omega_pen para que
    la comparacion sea limpia."""
    mods = list(iter_effective_weights(model))
    if not mods:
        raise RuntimeError("Sin modulos LoRA.")
    k = min(OMEGA_MODULES, len(mods))
    idx = torch.randperm(len(mods), generator=gen)[:k].tolist()
    total = sum(mods[i][1].float().norm(dim=1).var() for i in idx)
    return total / k


def _pen_dispatch(model, arm, gen):
    return rownorm_pen(model, gen) if arm == "rownorm" else omega_pen(model, arm, gen)


def _set_cos_sign(v):
    """COS_SIGN es global y omega_pen lo lee en cada llamada."""
    global COS_SIGN
    COS_SIGN = float(v)


def assert_connected(model, arm, gen):
    """La red de seguridad que no existia. Falla ruidosamente si no hay gradiente."""
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params: p.grad = None
    pen = _pen_dispatch(model, arm, gen)
    if not hasattr(pen, "grad_fn") or pen.grad_fn is None:
        raise RuntimeError("PENALIZACION DESCONECTADA (grad_fn=None). Busca un .data o .item().")
    pen.backward()
    g = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
    for p in params: p.grad = None
    if not torch.isfinite(g) or g.item() == 0.0:
        raise RuntimeError(f"GRADIENTE NULO ({g}). NO entrenes.")
    print(f"  [{arm}] conectada. valor={pen.item():+.4e}  ||grad||={g.item():.4e}")
    return g.item()


def calibrate_lambda(model, batch, arm, gen, target=None, n=5):
    if target is None:
        target = float(os.environ.get("OMEGA_TARGET", "0.1"))
    """El OMEGA_LAMBDA=0.05 anterior se ajusto contra una constante. No vale."""
    params = [p for p in model.parameters() if p.requires_grad]
    ratios = []
    for _ in range(n):
        for p in params: p.grad = None
        model(**batch).loss.backward()
        g_ce = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
        for p in params: p.grad = None
        _pen_dispatch(model, arm, gen).backward()
        g_om = torch.sqrt(sum((p.grad**2).sum() for p in params if p.grad is not None))
        ratios.append((g_om / (g_ce + 1e-12)).item())
    for p in params: p.grad = None
    lam = target / (statistics.mean(ratios) + 1e-12)
    print(f"  [{arm}] ratio grad={statistics.mean(ratios):.3e} -> lambda={lam:.4e}")
    return lam


def compute_fisher(model, loader, n_batches=32):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval(); i = -1
    for i, b in enumerate(loader):
        if i >= n_batches: break
        model.zero_grad()
        model(input_ids=b["input_ids"].to(DEVICE),
              labels=b["labels"].to(DEVICE)).loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher: fisher[n] /= max(min(i + 1, n_batches), 1)
    star = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    return fisher, star


def snapshot_star(model):
    """Copia de los pesos a proteger. SIN Fisher y SIN tocar datos: es la mitad
    barata de compute_fisher(), y estan separadas a proposito para que se vea
    en el codigo que L2-SP no necesita la tarea previa, solo la copia."""
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def l2sp_pen(model, star):
    """L2-SP: ewc_pen con la Fisher sustituida por unos. Ancla a los pesos de
    DESPUES de la tarea A, que es la capacidad que se quiere conservar."""
    return sum(((p - star[n]) ** 2).sum()
               for n, p in model.named_parameters() if n in star)


def ewc_pen(model, fisher, star):
    return sum((fisher[n] * (p - star[n]) ** 2).sum()
               for n, p in model.named_parameters() if n in fisher)


# ===========================================================================
# HUMANEVAL
# ===========================================================================
_SANDBOX = 'import sys, io, contextlib\n' \
           'src = open(sys.argv[1]).read()\n' \
           'buf = io.StringIO()\n' \
           'try:\n' \
           '    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n' \
           '        exec(src, {"__name__": "__main__"})\n' \
           '    print("PASS")\n' \
           'except BaseException:\n' \
           '    print("FAIL")\n'


def _run_sandboxed(code, timeout=10):
    """Ejecuta en SUBPROCESO. El exec() in-process del script original podia
    tumbar la corrida entera con el codigo generado por el modelo."""
    with tempfile.TemporaryDirectory() as d:
        cand = os.path.join(d, "cand.py"); runner = os.path.join(d, "run.py")
        open(cand, "w").write(code); open(runner, "w").write(_SANDBOX)
        try:
            r = subprocess.run([sys.executable, runner, cand], timeout=timeout,
                               capture_output=True, text=True)
            return "PASS" in r.stdout
        except Exception:
            return False


@torch.no_grad()
def humaneval(model, tok, n=HUMANEVAL_N, tag="", batch_size=None):
    """
    HumanEval pass@1 con generacion POR LOTES.

    El script original generaba de uno en uno: 164 problemas x 2 evaluaciones
    x ~10 s por generacion = ~55 min por celda, o sea el 70% del coste total.
    Por lotes de 32 baja a ~6 min. Mismo resultado, 8-10x mas rapido.

    El tokenizer usa padding_side="left", que es lo correcto para generar con
    modelos decoder-only: el padding queda ANTES del prompt y no contamina.
    """
    bs = batch_size or HE_BATCH
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    problems = [ds[i] for i in range(min(n, len(ds)))]
    passed, t0 = 0, time.time()
    model.eval()

    for start in range(0, len(problems), bs):
        chunk = problems[start:start + bs]
        enc = tok([p["prompt"] for p in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOK,
                             temperature=GEN_TEMP, do_sample=True,
                             pad_token_id=tok.eos_token_id)
        gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        for p, g in zip(chunk, gens):
            code = p["prompt"] + g + "\n" + p["test"] + "\ncheck(" + p["entry_point"] + ")\n"
            passed += _run_sandboxed(code)
        done = min(start + bs, len(problems))
        print("    [" + tag + "] " + str(done) + "/" + str(len(problems)) +
              " pass@1=" + format(passed/done, ".3f") +
              " (" + format(time.time()-t0, ".0f") + "s)")
    return passed / max(len(problems), 1)


# ===========================================================================
# DATOS / MODELO
# ===========================================================================
def load_model(seed, smoke):
    torch.manual_seed(seed); np.random.seed(seed)
    mid = "sshleifer/tiny-gpt2" if smoke else MODEL_ID
    tok = AutoTokenizer.from_pretrained(mid)
    tok.pad_token = tok.eos_token; tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16 if not smoke else torch.float32,
        low_cpu_mem_usage=True).to(DEVICE)
    if smoke:
        tgt = ["c_attn"]
    elif LORA_MODULES:
        tgt = LORA_MODULES
    else:
        tgt = LORA_TARGETS
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                     r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=tgt)
    pm = get_peft_model(m, cfg)
    # Cuantos modulos han recibido adaptador de verdad. Los dos brazos de una
    # comparacion de COLOCACION tienen que coincidir en este numero, o se
    # estaria comparando tamano de adaptador y no donde se pone.
    n_ad = sum(1 for nm, mod in pm.named_modules() if hasattr(mod, "lora_A"))
    print(f"  [lora] {n_ad} modulos adaptados"
          f"{' (colocacion explicita)' if LORA_MODULES else ' (por tipo)'}",
          flush=True)
    if LORA_MODULES and n_ad != len(LORA_MODULES):
        print(f"  [lora] AVISO: se pidieron {len(LORA_MODULES)} y se adaptaron "
              f"{n_ad}. Algun nombre no casa.", flush=True)
    return pm, tok


def _tok(tok, texts):
    enc = tok(texts, truncation=True, max_length=MAX_SEQ_LEN,
              padding="max_length", return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


def get_loader(tok, domain, smoke):
    n = 32 if smoke else MAX_SAMPLES
    if domain == "code":
        ds = load_dataset("code-search-net/code_search_net", "python", split="train",
                          trust_remote_code=True).select(range(n))
        def f(b):
            texts = ["### Docstring:\n" + d + "\n### Code:\n" + c
                     for d, c in zip(b["func_documentation_string"],
                                     b["whole_func_string"])]
            return _tok(tok, texts)
        ds = ds.map(f, batched=True, remove_columns=ds.column_names)
    else:
        # STREAMING: openwebtext son ~55 GB si se descarga entero y solo
        # necesitamos MAX_SAMPLES muestras. Con streaming no toca disco.
        it = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                          trust_remote_code=True).take(n)
        rows = [r["text"] for r in it]
        ds = Dataset.from_dict({"text": rows})
        ds = ds.map(lambda b: _tok(tok, b["text"]), batched=True,
                    remove_columns=ds.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


# ===========================================================================
# ENTRENAMIENTO
# ===========================================================================
def train_domain(model, loader, arm, wd, lam=None, gen=None, fisher=None, star=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=wd if arm == "wd" else 0.0)
    model.train()
    for step, b in enumerate(loader):
        loss = model(input_ids=b["input_ids"].to(DEVICE),
                     labels=b["labels"].to(DEVICE)).loss / GRAD_ACCUM
        if (arm.startswith("omega") or arm.startswith("cos")) and step % OMEGA_EVERY_K == 0:
            loss = loss + lam * omega_pen(model, arm, gen) / GRAD_ACCUM
        if arm == "rownorm" and step % OMEGA_EVERY_K == 0:
            loss = loss + lam * rownorm_pen(model, gen) / GRAD_ACCUM
        if arm == "ewc" and fisher is not None:
            loss = loss + EWC_LAMBDA * ewc_pen(model, fisher, star) / GRAD_ACCUM
        if arm == "l2sp" and star is not None:
            loss = loss + L2SP_LAMBDA * l2sp_pen(model, star) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
    return model


def run_cell(seed, arm, wd, smoke, he_n):
    t0 = time.time()
    print("\n" + "="*66 + "\nsemilla=" + str(seed) + "  brazo=" + arm +
          "  wd=" + str(wd) + "\n" + "="*66)
    model, tok = load_model(seed, smoke)
    code_tr  = get_loader(tok, "code", smoke)
    prose_tr = get_loader(tok, "prose", smoke)

    # RNG DEDICADO. Sin esto, la penalizacion consume el flujo global y el
    # brazo deja de ser comparable con el baseline: era el confound original.
    gen = torch.Generator().manual_seed(seed + 10_000)

    lam = None
    if arm.startswith("omega") or arm == "rownorm" or arm.startswith("cos"):
        assert_connected(model, arm, gen)
        b0 = next(iter(code_tr))
        lam = calibrate_lambda(model, {"input_ids": b0["input_ids"].to(DEVICE),
                                       "labels": b0["labels"].to(DEVICE)}, arm, gen)

    model = train_domain(model, code_tr, arm, wd, lam, gen)
    he_A = humaneval(model, tok, he_n, "tras A")
    print("  HumanEval pass@1 tras tarea A: " + format(he_A, ".4f"))
    ppl_A = perplexity(model, tok, "tras A")

    if arm == "omega_minv":
        os.environ["OMEGA_M_INV"] = "1"
        assert_minv_disponible()
    elif arm == "omega_lib":
        # explicito a proposito: que no lo herede del entorno por accidente
        os.environ["OMEGA_M_INV"] = "0"

    fisher = star = None
    if arm == "ewc":
        fisher, star = compute_fisher(model, code_tr)
    elif arm == "l2sp":
        # Solo la copia. Ni un batch de la tarea previa: esa es toda la gracia.
        star = snapshot_star(model)

    model = train_domain(model, prose_tr, arm, wd, lam, gen, fisher, star)
    he_B = humaneval(model, tok, he_n, "tras B")
    print("  HumanEval pass@1 tras tarea B: " + format(he_B, ".4f"))
    ppl_B = perplexity(model, tok, "tras B")

    ret = he_B / he_A if he_A > 0 else float("nan")
    print("  RETENCION: " + format(ret, ".4f") +
          "   (" + format(time.time()-t0, ".0f") + "s)")

    return dict(seed=seed, arm=arm, wd=wd, omega_lambda=lam,
                cos_sign=COS_SIGN if arm.startswith("cos") else None,
                omega_target=float(os.environ.get("OMEGA_TARGET", "0.1"))
                             if (arm.startswith("omega") or arm=="rownorm"
                                 or arm.startswith("cos")) else None,
                humaneval_after_A=he_A, humaneval_after_B=he_B,
                m_orientation=("1/lambda_2" if arm == "omega_minv" else
                               ("lambda_2" if arm == "omega_lib" else None)),
                ppl_after_A=ppl_A, ppl_after_B=ppl_B,
                ppl_delta=ppl_B - ppl_A,
                retention_pct=ret, seconds=time.time() - t0, smoke=smoke)



# ===========================================================================
# FASE 1a: cribado de vitalidad del canal de clustering coseno
# ===========================================================================
def _diag_modules_names(model):
    """DIAG_K modulos fijos y reproducibles: las primeras apariciones, que son
    las capas iniciales, donde el exceso medido es mayor."""
    return [nm for nm, _ in list(iter_effective_weights(model))[:DIAG_K]]


def _diag_snapshot(model, names):
    eff = dict(iter_effective_weights(model))
    return {nm: cosine_excess_CD_exact(eff[nm]) for nm in names if nm in eff}


def train_phase1a(model, loader, arm, lam, gen):
    """Solo tarea A, PHASE1A_STEPS pasos, sin wd ni ewc. Traza C/D EXACTO cada
    DIAG_EVERY pasos y la perdida de A. Devuelve (dC, lossA_final, traza).
    Si lam es 0 o None no se aplica penalizacion: sirve de referencia 'none'."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    names = _diag_modules_names(model)
    first = _diag_snapshot(model, names)
    trace = [dict(step=0, **first)]
    lossA = []

    step = 0
    for b in loader:
        if step >= PHASE1A_STEPS:
            break
        out = model(input_ids=b["input_ids"].to(DEVICE),
                    labels=b["labels"].to(DEVICE))
        loss = out.loss / GRAD_ACCUM
        lossA.append(out.loss.item())
        if lam and step % OMEGA_EVERY_K == 0:
            loss = loss + lam * omega_pen(model, arm, gen) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if step > 0 and step % DIAG_EVERY == 0:
            trace.append(dict(step=step, **_diag_snapshot(model, names)))
            print("    paso " + str(step) + "  C/D " +
                  "  ".join(format(trace[-1][nm], ".4f") for nm in names), flush=True)
        step += 1

    last = trace[-1]
    deltas = [abs(last[nm] - first[nm]) / (abs(first[nm]) + 1e-9) for nm in names]
    dC = sum(deltas) / max(1, len(deltas))
    lossA_final = sum(lossA[-20:]) / max(1, len(lossA[-20:]))
    return dC, lossA_final, trace


def run_phase1a(smoke, out):
    """Barrido 4 targets x 2 signos, 1 semilla, con la PUERTA prerregistrada.
    NO mide HumanEval: la pregunta es si dC se mueve, no cuanta retencion hay."""
    import os as _os
    targets = [float(x) for x in
               _os.environ.get("P1A_TARGETS", "0.003,0.01,0.03,0.1").split(",")]
    signs = [1.0, -1.0]
    seed = SEEDS[0]
    LEARN_MARGIN = float(_os.environ.get("LEARN_MARGIN", "0.15"))
    DC_ALIVE = float(_os.environ.get("DC_ALIVE", "0.02"))
    rows = []

    print("\n" + "#" * 66)
    print("FASE 1a: sobrevive el exceso coseno al entrenamiento con LoRA?")
    print("  " + str(len(targets)) + " targets x 2 signos, " +
          str(PHASE1A_STEPS) + " pasos, semilla " + str(seed))
    print("#" * 66)

    # referencia: perdida de A sin penalizacion
    print("\n### referencia none (sin penalizacion) ###")
    model, tok = load_model(seed, smoke)
    code_tr = get_loader(tok, "code", smoke)
    gen0 = torch.Generator().manual_seed(seed + 10_000)
    _, base_lossA, base_trace = train_phase1a(model, code_tr, "none", None, gen0)
    print("  loss A (none) = " + format(base_lossA, ".4f"))
    del model
    torch.cuda.empty_cache()

    for tg in targets:
        for sg in signs:
            _set_cos_sign(sg)
            _os.environ["OMEGA_TARGET"] = str(tg)
            arm = "cos_full"
            print("\n### target=" + str(tg) + "  signo=" + format(sg, "+.0f") + " ###")
            model, tok = load_model(seed, smoke)
            code_tr = get_loader(tok, "code", smoke)
            gen = torch.Generator().manual_seed(seed + 10_000)
            assert_connected(model, arm, gen)
            b0 = next(iter(code_tr))
            lam = calibrate_lambda(model, {"input_ids": b0["input_ids"].to(DEVICE),
                                           "labels": b0["labels"].to(DEVICE)},
                                   arm, gen, target=tg)
            dC, lossA, trace = train_phase1a(model, code_tr, arm, lam, gen)
            learned = bool(lossA <= base_lossA + LEARN_MARGIN)
            print("  dC=" + format(dC, ".4f") + "  lossA=" + format(lossA, ".4f") +
                  "  (none " + format(base_lossA, ".4f") + ")  aprende=" + str(learned))
            rows.append(dict(target=tg, sign=sg, lam=lam, dC=dC, lossA=lossA,
                             lossA_none=base_lossA, learned=learned, trace=trace))
            json.dump(dict(base_lossA=base_lossA, base_trace=base_trace, rows=rows),
                      open(out, "w"), indent=2)
            del model
            torch.cuda.empty_cache()

    # ------------------------- LA PUERTA -------------------------
    any_alive = any(r["dC"] >= DC_ALIVE for r in rows)
    any_learned = any(r["learned"] for r in rows)
    alive_and_learned = [r for r in rows if r["dC"] >= DC_ALIVE and r["learned"]]

    print("\n" + "=" * 66)
    print("PUERTA FASE 1a")
    print("  alguna config con dC >= " + str(DC_ALIVE) + ": " + str(any_alive))
    print("  alguna config aprende A: " + str(any_learned))
    print("  supervivientes (dC vivo Y aprende): " + str(len(alive_and_learned)))
    print("-" * 66)
    if not any_alive:
        print("  dC ~ 0 en todas. El exceso estatico del 36% NO sobrevive a")
        print("  LoRA. Desenlace 3: se cierra la linea, vale un parrafo (los")
        print("  estadisticos estaticos de W no predicen la estructura")
        print("  entrenable). NO pasar a 1b. NO montar merging.")
    elif not any_learned:
        print("  Ninguna config aprende A: el gradiente 20x desestabiliza.")
        print("  Desenlace 4: es reajuste de target o signo, no resultado.")
    else:
        print("  EL CANAL SOBREVIVE. Pasar a 1b con los supervivientes:")
        for r in alive_and_learned:
            print("    target=" + str(r["target"]) + " signo=" +
                  format(r["sign"], "+.0f") + "  dC=" + format(r["dC"], ".4f"))
        print("  Y el gancho de merging (interferencia entre deltas) pasa de")
        print("  idea a siguiente ronda con fundamento.")
    print("=" * 66)
    print("\nGuardado en " + out)
    return rows


# ===========================================================================
# ===========================================================================
# FASE 1b: que SIGNO ayuda a retener. Con HumanEval, no con dC.
# ===========================================================================
def run_phase1b(smoke, out, he_n, cell_idx=None):
    """2 signos x 2 semillas fuera de muestra, target fijo, retencion medida.

    Las semillas son NUEVAS (no estan en SEEDS) a proposito: el ajuste anterior
    uso 42 y 123, que si estan entre las diez de evaluacion, y eso produjo el
    artefacto none > wd por regresion a la media. Ajustar fuera de muestra
    handicapa al brazo nuevo, que es el sesgo conservador que interesa.

    cell_idx permite lanzar UNA celda por proceso y repartir entre GPUs con
    CUDA_VISIBLE_DEVICES: load_model hace .to("cuda") sobre un solo device, asi
    que 4 procesos en 4 GPUs corren las 4 celdas en paralelo.
    """
    import os as _os
    seeds = [int(x) for x in _os.environ.get("P1B_SEEDS", "7077,8088").split(",")]
    target = _os.environ.get("P1B_TARGET", "0.03")
    _os.environ["OMEGA_TARGET"] = target
    arm = _os.environ.get("P1B_ARM", "cos_full")

    CELLS = [(sg, sd) for sg in (1.0, -1.0) for sd in seeds]
    idxs = [cell_idx] if cell_idx is not None else list(range(len(CELLS)))

    print("\n" + "#" * 66)
    print("FASE 1b: que direccion del termino de clustering AYUDA a retener")
    print("  brazo=" + arm + "  target=" + target + "  semillas=" + str(seeds))
    print("  signo +1 BAJA C  |  signo -1 SUBE C (direccion del FSRI)")
    print("  celdas de este proceso: " + str(idxs) + " de " + str(len(CELLS)))
    print("#" * 66)

    rows = []
    for i in idxs:
        sg, sd = CELLS[i]
        _set_cos_sign(sg)
        print("\n### celda " + str(i) + ": signo " + format(sg, "+.0f") +
              "  semilla " + str(sd) + " ###")
        r = run_cell(sd, arm, 0.0, smoke, he_n)
        r["cell_idx"] = i
        rows.append(r)
        json.dump(rows, open(out, "w"), indent=2)

    # resumen solo si este proceso corrio todas las celdas
    if cell_idx is None and len(rows) == len(CELLS):
        print("\n" + "=" * 66)
        print("FASE 1b: RETENCION POR SIGNO Y SEMILLA")
        for r in rows:
            print("  signo " + format(r["cos_sign"], "+.0f") + "  semilla " +
                  str(r["seed"]) + "  retencion " +
                  format(r["retention_pct"], ".4f") +
                  "  (A=" + format(r["humaneval_after_A"], ".4f") +
                  " B=" + format(r["humaneval_after_B"], ".4f") + ")")
        print("-" * 66)
        # comparacion PAREADA por semilla: es el estadistico con potencia
        pares = []
        for sd in seeds:
            up = [r for r in rows if r["seed"] == sd and r["cos_sign"] < 0]
            dn = [r for r in rows if r["seed"] == sd and r["cos_sign"] > 0]
            if up and dn:
                d = up[0]["retention_pct"] - dn[0]["retention_pct"]
                pares.append(d)
                print("  semilla " + str(sd) + ": subir C menos bajar C = " +
                      format(d, "+.4f"))
        if pares:
            n_up = sum(1 for d in pares if d > 0)
            print("-" * 66)
            print("  subir C gana en " + str(n_up) + " de " + str(len(pares)) +
                  " semillas; diferencia media " +
                  format(sum(pares) / len(pares), "+.4f"))
            print("\n  LECTURA, y con 2 semillas es CRIBADO y no confirmacion:")
            print("  si gana subir C (signo -1), la direccion del FSRI se")
            print("  sostiene en LLM y la transferencia ecologica es real.")
            print("  Si gana bajar C (signo +1), la direccion del indice")
            print("  depende del dominio y hay que escribirlo en el FSRI.")
            print("  Si empatan, el canal se mueve pero no toca la retencion")
            print("  (desenlace 2) y lo que decide es el brazo compuesto.")
        print("=" * 66)
    print("\nGuardado en " + out)
    return rows


# ===========================================================================
# FASE 2: el compuesto coseno contra omega_lib, sobre las MISMAS 10 semillas
# ===========================================================================
def run_phase2(smoke, out, he_n, cell_idx=None):
    """Celdas = SEEDS x P2_ARMS, en el orden fijo de SEEDS.

    Por defecto solo cos_composite (10 celdas), porque las columnas de
    none/wd/ewc/omega_lib ya existen en la tabla de 10 semillas y el montaje
    es el mismo (q/v, target 0.03, EVERY_K=10). Si se quiere re-correr algun
    incumbente, pasarlo en P2_ARMS.

    cell_idx reparte entre GPUs con CUDA_VISIBLE_DEVICES, una celda por proceso.
    """
    import os as _os
    arms = _os.environ.get("P2_ARMS", "cos_composite").split(",")
    _os.environ.setdefault("OMEGA_TARGET", "0.03")
    CELLS = [(sd, ar) for sd in SEEDS for ar in arms]
    idxs = [cell_idx] if cell_idx is not None else list(range(len(CELLS)))

    print("\n" + "#" * 66)
    print("FASE 2: compuesto coseno frente a omega_lib (0.7662, 10 semillas)")
    print("  brazos=" + str(arms) + "  target=" + _os.environ["OMEGA_TARGET"] +
          "  C^s con s=" + str(COS_C_SIGN))
    print("  celdas de este proceso: " + str(idxs) + " de " + str(len(CELLS)))
    print("#" * 66)

    rows = []
    for i in idxs:
        sd, ar = CELLS[i]
        print("\n### celda " + str(i) + ": " + ar + "  semilla " + str(sd) + " ###")
        r = run_cell(sd, ar, 0.0, smoke, he_n)
        r["cell_idx"] = i
        rows.append(r)
        json.dump(rows, open(out, "w"), indent=2)
    print("\nGuardado en " + out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell",  action="store_true", help="una celda, para cronometrar")
    ap.add_argument("--all",   action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm",  default="omega_raw", choices=ARMS)
    ap.add_argument("--he-n", type=int, default=None)
    ap.add_argument("--phase1a", action="store_true",
                    help="cribado de vitalidad coseno: targets x signos, con puerta")
    ap.add_argument("--phase1b", action="store_true",
                    help="1b: que signo ayuda a retener, con HumanEval")
    ap.add_argument("--p1b-cell", type=int, default=None,
                    help="indice de celda 0-3 para repartir 1b entre GPUs")
    ap.add_argument("--phase2", action="store_true",
                    help="fase 2: compuesto coseno sobre las 10 semillas")
    ap.add_argument("--p2-cell", type=int, default=None,
                    help="indice de celda para repartir la fase 2 entre GPUs")
    ap.add_argument("--out",  default="rerun_results.json")
    ap.add_argument("--cells", default=None,
                    help="indices de celda separados por comas, p.ej. 0,3,7. "
                         "Lo usa launch_parallel.sh para repartir entre GPUs.")
    ap.add_argument("--best-wd", type=float, default=None,
                    help="salta la rejilla de weight decay y usa este valor")
    ap.add_argument("--he-batch", type=int, default=None)
    ap.add_argument("--omega-sweep", default=None,
                    help="targets de calibracion separados por comas, p.ej. 0.03,0.1,0.3,0.5")
    ap.add_argument("--omega-arm", default="omega_lib",
                    choices=["omega_raw","omega_lib","rownorm","omega_spectral",
                             "omega_minv"],
                    help="que brazo barrer con --omega-sweep")
    ap.add_argument("--l2sp-sweep", default=None,
                    help="barrido de lambda para L2-SP, p.ej. 0.1,1,10,100")
    ap.add_argument("--ewc-sweep", default=None,
                    help="lambdas de EWC separados por comas, p.ej. 100,500,1000,5000,10000. Corre EWC con cada uno en la semilla 42 para ajustarlo antes de la corrida grande.")
    a = ap.parse_args()
    if a.he_batch:
        global HE_BATCH
        HE_BATCH = a.he_batch

    he_n = a.he_n or (8 if a.smoke else HUMANEVAL_N)
    print(DECISION_RULE)
    rows = []

    if a.phase1a:
        run_phase1a(a.smoke, a.out)
        return

    if a.phase1b:
        run_phase1b(a.smoke, a.out, he_n, a.p1b_cell)
        return

    if a.phase2:
        run_phase2(a.smoke, a.out, he_n, a.p2_cell)
        return

    if a.omega_sweep is not None:
        # Barrido del objetivo de calibracion de omega (y rownorm). Ajustar
        # omega con la misma vara que wd y ewc: sin esto, la comparacion
        # favorece a quien se ajusta. target = fraccion del gradiente de la
        # tarea que ocupa la penalizacion (3%, 10%, 30%, 50%).
        import os as _os
        which = a.omega_arm
        targets = [float(x) for x in a.omega_sweep.split(",")]
        wd = a.best_wd if a.best_wd is not None else 0.0
        for tg in targets:
            _os.environ["OMEGA_TARGET"] = str(tg)
            print("\n### " + which + " target = " + str(tg) + " ###")
            r = run_cell(TUNE_SEED, which, wd, a.smoke, he_n)
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=2)
        best = max(rows, key=lambda r: r["retention_pct"])
        print("\n>>> Mejor " + which + ": target=" + str(best["omega_target"]) +
              " con retencion " + format(best["retention_pct"], ".4f"))
        print("Guardado en " + a.out)
        return

    if a.l2sp_sweep is not None:
        # Calcado del barrido de EWC. Ajusta sobre TUNE_SEED, disjunta de las
        # diez, y elige por CAPACIDAD ABSOLUTA y no por el cociente: un lambda
        # fuerte infla el cociente hundiendo el denominador.
        import os as _os
        lams = [float(x) for x in a.l2sp_sweep.split(",")]
        wd = a.best_wd if a.best_wd is not None else 0.0
        for lam in lams:
            _os.environ["L2SP_LAMBDA"] = str(lam)
            globals()["L2SP_LAMBDA"] = lam
            print("\n### L2-SP lambda = " + str(lam) + " ###")
            r = run_cell(TUNE_SEED, "l2sp", wd, a.smoke, he_n)
            r["l2sp_lambda"] = lam
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=2)
        best = max(rows, key=lambda r: r["humaneval_after_B"])
        print("\n>>> Mejor L2-SP por CAPACIDAD ABSOLUTA: lambda=" +
              str(best["l2sp_lambda"]) + " con " +
              format(best["humaneval_after_B"], ".4f"))
        print("Guardado en " + a.out)
        return

    if a.ewc_sweep is not None:
        # Barrido de EWC en una semilla para ajustar su lambda. EWC salio
        # rindiendo como "no hacer nada" (0.601 vs 0.605), lo que sugiere que
        # su lambda por defecto (1000) esta mal puesto. Esto lo comprueba.
        import os as _os
        lams = [float(x) for x in a.ewc_sweep.split(",")]
        wd = a.best_wd if a.best_wd is not None else 0.0
        for lam in lams:
            _os.environ["EWC_LAMBDA"] = str(lam)
            print("\n### EWC lambda = " + str(lam) + " ###")
            r = run_cell(TUNE_SEED, "ewc", wd, a.smoke, he_n)
            r["ewc_lambda"] = lam
            rows.append(r)
            json.dump(rows, open(a.out, "w"), indent=2)
        best = max(rows, key=lambda r: r["retention_pct"])
        print("\n>>> Mejor EWC: lambda=" + str(best["ewc_lambda"]) +
              " con retencion " + format(best["retention_pct"], ".4f"))
        print("Guardado en " + a.out)
        return

    if a.cells is not None:
        # Reparto para ejecucion paralela. El orden de CELLS es fijo y
        # deterministico, asi que cada GPU sabe cuales le tocan sin coordinarse.
        CELLS = [(sd, ar) for sd in SEEDS for ar in ARMS]
        wd = a.best_wd if a.best_wd is not None else 0.0
        for i in [int(x) for x in a.cells.split(",")]:
            sd, ar = CELLS[i]
            print("\n### celda " + str(i) + " de " + str(len(CELLS)) + " ###")
            rows.append(run_cell(sd, ar, wd, a.smoke, he_n))
            json.dump(rows, open(a.out, "w"), indent=2)
        json.dump(rows, open(a.out, "w"), indent=2)
        print("\nGuardado en " + a.out)
        return

    if a.all:
        print("\n### Paso 0: ajustar weight decay ###")
        wd_rows = [run_cell(TUNE_SEED, "wd", w, a.smoke, he_n) for w in WD_GRID]
        best_wd = max(wd_rows, key=lambda r: r["retention_pct"])["wd"]
        print("\n>>> weight decay ajustado: " + str(best_wd))
        rows += wd_rows
        for seed, arm in itertools.product(SEEDS, ARMS):
            rows.append(run_cell(seed, arm, best_wd, a.smoke, he_n))
            json.dump(rows, open(a.out, "w"), indent=2)   # guardar tras cada celda
    else:
        _wd = a.best_wd if a.best_wd is not None else 0.0
        rows.append(run_cell(a.seed, a.arm, _wd, a.smoke, he_n))

    json.dump(rows, open(a.out, "w"), indent=2)
    print("\nGuardado en " + a.out)
    if a.cell:
        s = rows[0]["seconds"]
        print("\nUna celda = " + format(s/60, ".1f") + " min. Rejilla completa "
              "(19 celdas) ~ " + format(19*s/3600, ".1f") + " h de GPU.")


if __name__ == "__main__":
    main()
