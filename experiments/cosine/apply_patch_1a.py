#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_patch_1a.py  --  aplica los edits 4 a 7 del parche de FASE 1a sobre
rerun_retention.py. Los edits 1-3 (ARMS, LORA_TARGETS, constantes) se dan por
hechos; si no lo estan, el script los aplica tambien.

Uso, desde la carpeta donde esta rerun_retention.py:
    python apply_patch_1a.py

Hace copia de seguridad, ancla por CONTENIDO (no por numero de linea, que se
desplaza), verifica con ast.parse y aborta sin escribir si algo no casa.
"""
import ast
import os
import shutil
import sys

P = "rerun_retention.py"

COSINE_CORE = '''
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

'''

OMEGA_PEN_OLD = '''def omega_pen(model, arm, gen):
    mods = list(iter_effective_weights(model))
    if not mods:
        raise RuntimeError("Sin modulos LoRA. Revisa LORA_TARGETS.")
    k = min(OMEGA_MODULES, len(mods))
    idx = torch.randperm(len(mods), generator=gen)[:k].tolist()
    if arm == "omega_raw":
        total = sum(hutchinson_tr_a3(mods[i][1], OMEGA_PROBES, gen) for i in idx)
    else:
        core = omega_lib_core()
        total = sum(core(mods[i][1]) for i in idx)
    return total / k'''

OMEGA_PEN_NEW = '''def omega_pen(model, arm, gen):
    mods = list(iter_effective_weights(model))
    if not mods:
        raise RuntimeError("Sin modulos LoRA. Revisa LORA_TARGETS.")
    k = min(OMEGA_MODULES, len(mods))
    idx = torch.randperm(len(mods), generator=gen)[:k].tolist()
    if arm == "omega_raw":
        total = sum(hutchinson_tr_a3(mods[i][1], OMEGA_PROBES, gen) for i in idx)
    elif arm in ("cos_full", "cos_noCoex"):
        # FASE 1a: SOLO el termino de clustering, +-log(Tr(A^3)_coseno).
        # Con el log-ratio compuesto el signo no se aisla; aqui si, y la
        # pregunta de 1a es exactamente si este canal esta vivo y mueve dC.
        # COS_SIGN=+1 sube C (direccion del objetivo actual y del FSRI),
        # COS_SIGN=-1 la baja (la prediccion mecanica opuesta).
        total = sum(COS_SIGN * torch.log(
            cosine_tr_a3_hutch(mods[i][1], OMEGA_PROBES, gen) + 1e-6)
            for i in idx)
    else:
        core = omega_lib_core()
        total = sum(core(mods[i][1]) for i in idx)
    return total / k'''

TRAIN_OLD = '        if arm.startswith("omega") and step % OMEGA_EVERY_K == 0:'
TRAIN_NEW = '        if (arm.startswith("omega") or arm.startswith("cos")) and step % OMEGA_EVERY_K == 0:'

DISPATCH_OLD = '''def _pen_dispatch(model, arm, gen):
    return rownorm_pen(model, gen) if arm == "rownorm" else omega_pen(model, arm, gen)'''

DISPATCH_NEW = '''def _pen_dispatch(model, arm, gen):
    return rownorm_pen(model, gen) if arm == "rownorm" else omega_pen(model, arm, gen)


def _set_cos_sign(v):
    """COS_SIGN es global y omega_pen lo lee en cada llamada."""
    global COS_SIGN
    COS_SIGN = float(v)'''

PHASE1A_BLOCK = '''
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

    print("\\n" + "#" * 66)
    print("FASE 1a: sobrevive el exceso coseno al entrenamiento con LoRA?")
    print("  " + str(len(targets)) + " targets x 2 signos, " +
          str(PHASE1A_STEPS) + " pasos, semilla " + str(seed))
    print("#" * 66)

    # referencia: perdida de A sin penalizacion
    print("\\n### referencia none (sin penalizacion) ###")
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
            print("\\n### target=" + str(tg) + "  signo=" + format(sg, "+.0f") + " ###")
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

    print("\\n" + "=" * 66)
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
    print("\\nGuardado en " + out)
    return rows

'''

ARG_OLD = '    ap.add_argument("--he-n", type=int, default=None)'
ARG_NEW = '''    ap.add_argument("--he-n", type=int, default=None)
    ap.add_argument("--phase1a", action="store_true",
                    help="cribado de vitalidad coseno: targets x signos, con puerta")'''

DISPATCH_MAIN_OLD = '''    he_n = a.he_n or (8 if a.smoke else HUMANEVAL_N)
    print(DECISION_RULE)
    rows = []'''

DISPATCH_MAIN_NEW = '''    he_n = a.he_n or (8 if a.smoke else HUMANEVAL_N)
    print(DECISION_RULE)
    rows = []

    if a.phase1a:
        run_phase1a(a.smoke, a.out)
        return'''


def main():
    if not os.path.exists(P):
        sys.exit("No encuentro " + P + " en este directorio.")
    src = open(P).read()

    # --- edits 1-3, por si no estan ---
    if '"cos_full"' not in src:
        src = src.replace(
            'ARMS     = ["none", "wd", "ewc", "rownorm", "omega_raw", "omega_lib"]',
            'ARMS     = ["none", "wd", "ewc", "rownorm", "omega_raw", "omega_lib", "cos_full", "cos_noCoex"]')
        print("edit 1 aplicado (ARMS)")
    else:
        print("edit 1 ya estaba")

    if 'LORA_TARGETS = ["q_proj", "v_proj"]' in src:
        src = src.replace('LORA_TARGETS = ["q_proj", "v_proj"]',
                          'LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]')
        print("edit 2 aplicado (LORA_TARGETS)")
    else:
        print("edit 2 ya estaba")

    if "COS_SIGN" not in src:
        anchor = 'EWC_LAMBDA    = float(os.environ.get("EWC_LAMBDA", "1000.0"))  # barrible'
        if anchor not in src:
            sys.exit("ABORTO: no encuentro el ancla de EWC_LAMBDA para el edit 3")
        src = src.replace(anchor, anchor + '''
# --- Fase 1a: construccion coseno y diagnostico ---
COS_SIGN     = float(os.environ.get("COS_SIGN", "1.0"))
DIAG_EVERY   = int(os.environ.get("DIAG_EVERY", "25"))
DIAG_K       = 4
PHASE1A_STEPS = int(os.environ.get("PHASE1A_STEPS", "200"))''')
        print("edit 3 aplicado (constantes)")
    else:
        print("edit 3 ya estaba")

    # --- edit 4: nucleo coseno, antes de _LIB_CORE ---
    if "def cosine_A(" in src:
        print("edit 4 ya estaba")
    else:
        anchor = "_LIB_CORE = None"
        if anchor not in src:
            sys.exit("ABORTO: no encuentro _LIB_CORE para el edit 4")
        src = src.replace(anchor, COSINE_CORE + "\n" + anchor, 1)
        print("edit 4 aplicado (nucleo coseno + gancho merging)")

    # --- edit 5: omega_pen ---
    if "cos_full" in src and "cosine_tr_a3_hutch(mods[i][1]" in src:
        print("edit 5 ya estaba")
    else:
        if OMEGA_PEN_OLD not in src:
            sys.exit("ABORTO: omega_pen no coincide literalmente. Revisar a mano.")
        src = src.replace(OMEGA_PEN_OLD, OMEGA_PEN_NEW, 1)
        print("edit 5 aplicado (omega_pen enruta cos)")

    # --- edit 5b: _set_cos_sign ---
    if "_set_cos_sign" in src:
        print("edit 5b ya estaba")
    else:
        if DISPATCH_OLD not in src:
            sys.exit("ABORTO: _pen_dispatch no coincide. Revisar a mano.")
        src = src.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
        print("edit 5b aplicado (_set_cos_sign)")

    # --- edit 6: train_domain ---
    if TRAIN_NEW.strip() in src:
        print("edit 6 ya estaba")
    else:
        if TRAIN_OLD not in src:
            sys.exit("ABORTO: la linea de train_domain no coincide. Revisar a mano.")
        src = src.replace(TRAIN_OLD, TRAIN_NEW, 1)
        print("edit 6 aplicado (train_domain enruta cos)")

    # --- bloque phase1a, antes de def main ---
    if "def run_phase1a(" in src:
        print("bloque phase1a ya estaba")
    else:
        anchor = "# ===========================================================================\ndef main():"
        if anchor in src:
            src = src.replace(anchor, PHASE1A_BLOCK + "\n" + anchor, 1)
        elif "\ndef main():" in src:
            src = src.replace("\ndef main():", PHASE1A_BLOCK + "\ndef main():", 1)
        else:
            sys.exit("ABORTO: no encuentro def main() para insertar run_phase1a")
        print("bloque phase1a insertado")

    # --- edit 7: flag y dispatch en main ---
    if "--phase1a" in src:
        print("edit 7a ya estaba")
    else:
        if ARG_OLD not in src:
            sys.exit("ABORTO: no encuentro el argumento --he-n para el edit 7")
        src = src.replace(ARG_OLD, ARG_NEW, 1)
        print("edit 7a aplicado (flag --phase1a)")

    if "if a.phase1a:" in src:
        print("edit 7b ya estaba")
    else:
        if DISPATCH_MAIN_OLD not in src:
            sys.exit("ABORTO: no encuentro el arranque de main para el edit 7b")
        src = src.replace(DISPATCH_MAIN_OLD, DISPATCH_MAIN_NEW, 1)
        print("edit 7b aplicado (dispatch en main)")

    # --- verificacion antes de escribir ---
    try:
        ast.parse(src)
    except SyntaxError as e:
        sys.exit("ABORTO: el resultado no compila: " + str(e))

    if not os.path.exists(P + ".prepatch"):
        shutil.copy(P, P + ".prepatch")
        print("copia de seguridad: " + P + ".prepatch")
    open(P, "w").write(src)
    print("\nOK. " + P + " parcheado y verificado (ast.parse limpio).")
    print("Siguiente: python experiments/rerun_retention.py --phase1a --smoke --out p1a_smoke.json")
    print("           (desde /workspace/omega-s, para que el import de omega_s funcione)")


if __name__ == "__main__":
    main()
