#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp6b_flops_by_form.py  --  FLOPs por FORMA de la penalizacion.

exp6_wd_vs_omega_pruning.py mide 54.46% de reduccion de FLOPs con
"Omega-S + group lasso", pero la penalizacion que usa es la CRUDA,
Tr((W W^T)^3), no el compuesto log-ratio que produce los resultados de
retencion. Son dos objetivos distintos bajo el mismo nombre.

Este script contesta dos preguntas que exp6 no puede contestar:

  1. Cuanto aporta Omega POR ENCIMA de group lasso solo?
     exp6 no tiene el brazo de group lasso aislado, asi que su 54.46% es una
     propiedad del pipeline combinado, no de Omega.

  2. Depende del resultado de QUE FORMA se use?
     Hay una prediccion falsable: la forma cruda es homogenea de grado 6 en W,
     luego su gradiente escala como c^5. Verificado numericamente: a un cuarto
     de escala conserva el 0.1% de su fuerza. Y el group lasso ENCOGE pesos por
     diseno. Asi que la forma cruda deberia apagarse sola justo en este
     montaje, mientras que el compuesto (sigmoid acotado + logaritmo) no.
     Si la prediccion es correcta, gl+compuesto deberia separarse de gl solo
     mas que gl+crudo.

BRAZOS: none | wd (barrido) | gl solo | gl+crudo | gl+compuesto | gl+coseno

CALIBRACION: las tres formas tienen escalas de valor incomparables (la cruda
es de grado 6 y llega a 1e14, el compuesto es un logaritmo del orden de 10).
Compararlas a lambda fijo no significa nada. Aqui lambda se CALIBRA por brazo
para que la norma del gradiente de la penalizacion sea una fraccion fija de la
del gradiente de la tarea, que es el mismo protocolo de los experimentos de
retencion.

CRIBADO PREVIO CON PUERTA: antes de gastar tiempo se comprueba, sobre los
pesos ENTRENADOS del MLP, si la construccion coseno descomprime de verdad el
canal de clustering ahi. En Llama lo hace; en un MLP pequeno entrenado desde
cero puede no hacerlo, y entonces el brazo coseno no tiene nada que probar.
Ojo ademas: la capa 256x128 no es cuadrada pero 256x256 si lo seria en otras
arquitecturas, asi que la bifurcacion por forma de la libreria tambien aparece
aqui y se declara en la salida.

USO:
    python exp6b_flops_by_form.py            # completo
    python exp6b_flops_by_form.py --smoke    # 1 epoca, para verificar que corre

NOTA: no se ha podido ejecutar en el entorno donde se escribio (sin torch ni
MNIST). Correr primero con --smoke.
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reutiliza la maquinaria ya validada de exp6: modelo, datos, poda y FLOPs
from exp6_wd_vs_omega_pruning import (      # noqa: E402
    DEVICE, SEED, EPOCHS, LR, WD_GRID, GL_LAMBDA, ZERO_THRESHOLD,
    get_loader, build_model, get_linear_layers, group_lasso_penalty,
    train_wd, run_pruning_pipeline, eval_acc,
)

EPS = 1e-6
# Semillas de AJUSTE fuera del conjunto de evaluacion, misma convencion que los
# experimentos de retencion: seleccionar el hiperparametro sobre las semillas en
# las que luego se reporta es lo que produjo el artefacto de none sobre wd.
TUNE_SEEDS = [7077, 8088]
EVAL_SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
TARGET = 0.03          # fraccion del gradiente de la tarea, como en retencion
EVERY_K = 10
PROBES = 16


# ==========================================================================
# Las tres formas de la penalizacion
# ==========================================================================
def _hutch_tr_a3(A, probes, gen):
    n = A.shape[0]
    tot = 0.0
    for _ in range(probes):
        z = (torch.randint(0, 2, (n, 1), generator=gen,
                           dtype=torch.float32) * 2.0 - 1.0).to(A.device)
        tot = tot + (z.t() @ (A @ (A @ (A @ z)))).squeeze()
    return tot / probes


def pen_raw(W, gen):
    """Forma 1: Tr((W W^T)^3) crudo. Homogenea de grado 6 en W."""
    A = W @ W.t()
    return _hutch_tr_a3(A, PROBES, gen)


def _composite(A, gen):
    """log((M*Coex)/(C*D)) sobre una A ya construida. Replica de
    StochasticOmegaS: mismas formulas, mismo epsilon, misma iteracion."""
    n = A.shape[0]
    D = torch.mean(A)
    k = torch.sum(A, dim=1)
    Coex = torch.var(k) + EPS
    C = _hutch_tr_a3(A, PROBES, gen) / (torch.norm(A, p="fro") ** 3 + EPS) + EPS
    v = torch.randn((n, 1), generator=gen, dtype=torch.float32).to(A.device)
    v = v - torch.mean(v)
    v = v / (torch.norm(v) + EPS)
    mx = torch.max(k)
    for _ in range(3):
        Lv = (k.view(-1, 1) * v) - A @ v
        v = (2 * mx) * v - Lv
        v = (v - torch.mean(v)) / (torch.norm(v) + EPS)
    M = torch.abs((v.t() @ ((k.view(-1, 1) * v) - A @ v)).squeeze()) + EPS
    return torch.log((M * Coex) / (C * D + EPS))


def pen_lib(W, gen):
    """Forma 2: compuesto con la construccion de la libreria, incluida su
    bifurcacion por forma (Gram si no es cuadrada, W directo si lo es)."""
    Wc = W @ W.t() if W.shape[0] != W.shape[1] else W
    S = torch.sigmoid(torch.abs(Wc))
    return _composite((S + S.t()) / 2.0, gen)


def pen_cos(W, gen, eps=1e-8):
    """Forma 3: compuesto con construccion coseno por filas, diagonal a cero."""
    Wn = W / (W.norm(dim=1, keepdim=True) + eps)
    A = (Wn @ Wn.t()).abs().clamp(0.0, 1.0)
    A = A - torch.diag(torch.diagonal(A))
    return _composite(A, gen)


def pen_rownorm(W, gen):
    """Control de mecanismo: varianza de las normas de fila. Cero topologia.
    Misma definicion que rownorm_pen en experiments/rerun_retention.py, para
    que el control sea el mismo que en los experimentos de retencion."""
    return W.float().norm(dim=1).var()


PENALTIES = {"raw": pen_raw, "lib": pen_lib, "cos": pen_cos,
             "rownorm": pen_rownorm}


# ==========================================================================
# Cribado previo con puerta: descomprime el coseno sobre ESTOS pesos?
# ==========================================================================
def excess_CD(A):
    A = A.double().clone()
    A.fill_diagonal_(0.0)
    n = A.shape[0]
    A2 = A @ A
    C = (A * A2).sum() / (A2.sum() - torch.diagonal(A2).sum() + 1e-12)
    D = (A.sum() - torch.diagonal(A).sum()) / (n * (n - 1) + 1e-12)
    return float(C / (D + 1e-12))


def cribado(model):
    """Sobre los pesos ENTRENADOS. 1.0 = sin exceso triadico."""
    print("\n" + "=" * 74)
    print("CRIBADO PREVIO: descomprime la construccion coseno en ESTE modelo?")
    print("=" * 74)
    print(f"{'capa':22} {'forma':>12} {'sigmoid':>10} {'coseno':>10}")
    filas = []
    for nm, l in get_linear_layers(model):
        W = l.weight.detach().float()
        cuadrada = W.shape[0] == W.shape[1]
        Wc = W if cuadrada else W @ W.t()
        S = torch.sigmoid(torch.abs(Wc))
        e_sig = excess_CD((S + S.t()) / 2.0)
        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        Ac = (Wn @ Wn.t()).abs().clamp(0, 1)
        e_cos = excess_CD(Ac)
        rama = "cuadrada" if cuadrada else "Gram"
        print(f"{nm:22} {rama:>12} {e_sig:10.4f} {e_cos:10.4f}")
        filas.append(dict(capa=nm, rama=rama, sigmoid=e_sig, coseno=e_cos))
    max_cos = max(f["coseno"] for f in filas)
    vivo = max_cos > 1.02
    print("-" * 74)
    print(f"  maximo exceso con coseno: {max_cos:.4f}")
    if not vivo:
        print("  PUERTA: el coseno NO descomprime sobre estos pesos. El brazo")
        print("  'cos' no tiene canal que probar y se OMITE. No es un fallo:")
        print("  es que la estructura angular de un MLP pequeno entrenado desde")
        print("  cero no se parece a la de Llama.")
    else:
        print("  PUERTA: canal vivo, el brazo 'cos' se corre.")
    return vivo, filas


# ==========================================================================
# Entrenamiento con calibracion por ratio
# ==========================================================================
def calibrar(model, batch, forma, gen):
    """lambda tal que ||lambda * grad Omega|| = TARGET * ||grad CE||."""
    x, y = batch
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    g_ce = sum(p.grad.pow(2).sum().item() for p in model.parameters()
               if p.grad is not None) ** 0.5
    model.zero_grad()
    pen = sum(PENALTIES[forma](l.weight, gen) for _, l in get_linear_layers(model))
    pen.backward()
    g_om = sum(p.grad.pow(2).sum().item() for p in model.parameters()
               if p.grad is not None) ** 0.5
    model.zero_grad()
    if g_om < 1e-30:
        print(f"    AVISO: gradiente de la penalizacion ~0 ({g_om:.2e}); "
              f"la forma '{forma}' esta inerte en la inicializacion")
        return 0.0, g_ce, g_om
    return TARGET * g_ce / g_om, g_ce, g_om


def train_arm(loader, forma=None, usar_gl=True, epochs=EPOCHS, semilla=SEED):
    """forma=None -> sin penalizacion topologica. usar_gl controla group lasso."""
    torch.manual_seed(semilla)
    model = build_model()
    gen = torch.Generator().manual_seed(semilla + 10_000)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lam = None
    step = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if forma is not None and lam is None:
                lam, g_ce, g_om = calibrar(model, (x, y), forma, gen)
                print(f"    calibracion: ||grad CE||={g_ce:.3e}  "
                      f"||grad Omega||={g_om:.3e}  ->  lambda={lam:.4e}")
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if usar_gl:
                loss = loss + GL_LAMBDA * group_lasso_penalty(model)
            if forma is not None and lam and step % EVERY_K == 0:
                loss = loss + lam * sum(PENALTIES[forma](l.weight, gen)
                                        for _, l in get_linear_layers(model))
            loss.backward()
            opt.step()
            step += 1
    return model


# ==========================================================================
def train_exp6_exacto(loader, epochs, semilla=SEED):
    """La configuracion LITERAL de exp6: penalizacion CRUDA con OMEGA_LAMBDA
    fijo (no calibrada) y group lasso. Es el ancla: si esto no reproduce el
    54.46% con ~95% de accuracy, cualquier comparacion posterior es sobre otro
    regimen y no dice nada del resultado publicado."""
    from exp6_wd_vs_omega_pruning import OMEGA_LAMBDA, omega_penalty
    torch.manual_seed(semilla)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    step = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss = loss + GL_LAMBDA * group_lasso_penalty(model)
            if step % 10 == 0:
                loss = loss + OMEGA_LAMBDA * omega_penalty(model)
            loss.backward()
            opt.step()
            step += 1
    return model


def ratio_efectivo(model, batch, forma, gen, lam_fijo):
    """Que fraccion del gradiente de la tarea representa un lambda FIJO dado.
    Sirve para trasladar el punto de operacion del ancla a las demas formas."""
    x, y = batch
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    g_ce = sum(p.grad.pow(2).sum().item() for p in model.parameters()
               if p.grad is not None) ** 0.5
    model.zero_grad()
    sum(PENALTIES[forma](l.weight, gen) for _, l in get_linear_layers(model)).backward()
    g_om = sum(p.grad.pow(2).sum().item() for p in model.parameters()
               if p.grad is not None) ** 0.5
    model.zero_grad()
    return lam_fijo * g_om / (g_ce + 1e-30), g_ce, g_om


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target", type=float, default=None,
                    help="si se omite, se DERIVA del ancla exp6")
    ap.add_argument("--max-caida", type=float, default=1.0,
                    help="puntos de accuracy que puede perder la poda para que "
                         "el brazo cuente como comparable")
    ap.add_argument("--modo", choices=["tune","eval","todo"], default="todo",
                    help="tune = rejilla en semillas fuera de muestra; "
                         "eval = configuraciones fijas en las diez semillas")
    ap.add_argument("--configs", default="exp6b_configs.json",
                    help="donde se guardan/leen las configuraciones de frontera")
    ap.add_argument("--rapido", action="store_true",
                    help="rejilla reducida para una primera pasada")
    ap.add_argument("--out", default="exp6b_flops_by_form.json")
    a = ap.parse_args()
    ep = 1 if a.smoke else EPOCHS

    print(f"Dispositivo: {DEVICE}   epocas: {ep}")
    tr, te = get_loader(train=True), get_loader(train=False)

    # ---------------------------------------------------------------- ANCLA
    print("\n" + "=" * 74)
    print("ANCLA: reproducir exp6 exacto (crudo, OMEGA_LAMBDA fijo, sin calibrar)")
    print("=" * 74)
    print("  esperado: acc ~95%, sin perdida al podar, reduccion de FLOPs ~54%")
    m_anc = train_exp6_exacto(tr, ep)
    acc_anc = eval_acc(m_anc, te)
    pr_anc, ffa, fpa = run_pruning_pipeline(m_anc, "ancla")
    acc_anc_p = eval_acc(pr_anc, te)
    red_anc = 1.0 - fpa / ffa
    caida_anc = 100 * (acc_anc - acc_anc_p)
    print(f"  obtenido: acc {acc_anc:.4%} -> {acc_anc_p:.4%} "
          f"(caida {caida_anc:+.2f} pp)  FLOPs -{red_anc:.2%}")

    reproduce = (abs(red_anc - 0.5446) < 0.10) and (caida_anc < a.max_caida)
    if reproduce:
        print("  ANCLA OK: el montaje reproduce el resultado publicado.")
    else:
        print("  *** EL ANCLA NO REPRODUCE ***")
        print(f"      FLOPs {red_anc:.2%} frente al 54.46% publicado, "
              f"caida de accuracy {caida_anc:+.2f} pp")
        print("      Las comparaciones de abajo son sobre OTRO regimen y NO")
        print("      dicen nada sobre el numero del paper. Ajustar GL_LAMBDA o")
        print("      las epocas hasta que el ancla cuadre, y volver a correr.")

    # punto de operacion: que ratio representa el lambda fijo del ancla
    gen0 = torch.Generator().manual_seed(SEED + 1)
    b0 = next(iter(tr)); b0 = (b0[0].to(DEVICE), b0[1].to(DEVICE))
    from exp6_wd_vs_omega_pruning import OMEGA_LAMBDA as LAM_FIJO
    r_anc, g_ce, g_om = ratio_efectivo(m_anc, b0, "raw", gen0, LAM_FIJO)
    target = a.target if a.target is not None else r_anc
    print(f"\n  punto de operacion del ancla: lambda fijo {LAM_FIJO} equivale a")
    print(f"  un ratio de {r_anc:.4e} del gradiente de la tarea "
          f"(||grad CE||={g_ce:.2e}, ||grad Omega||={g_om:.2e})")
    print(f"  -> las demas formas se calibran a target = {target:.4e}")
    globals()["TARGET"] = target

    # el cribado necesita un modelo ENTRENADO: se usa el de group lasso solo
    print("\n### brazo base: group lasso solo (es tambien el control que faltaba)")
    m_gl = train_arm(tr, forma=None, usar_gl=True, epochs=ep)
    vivo, filas_cribado = cribado(m_gl)

    # ---------------------------------------------------------------- BARRIDO
    # Comparar en UN punto de fuerza no vale: la misma forma cruda da 54.46% a
    # ratio 36.8 y 93.30% a ratio 1, con la accuracy intacta en ambos. Hay que
    # medir la FRONTERA: mayor reduccion de FLOPs con el modelo intacto.
    #
    # Y hay que separar AJUSTE de EVALUACION. Seleccionar la configuracion sobre
    # las mismas semillas en las que luego se reporta infla el resultado; es lo
    # que produjo el artefacto de none sobre wd en los experimentos de retencion.
    #
    # EWC NO APLICA: necesita la Fisher de una tarea ANTERIOR y esto es de tarea
    # unica. Sin secuencia A->B no hay Fisher que calcular.
    from exp6_wd_vs_omega_pruning import GL_LAMBDA as GL0
    import exp6_wd_vs_omega_pruning as E6

    GRID_RATIO = [0.3, 1.0, 3.0, 10.0, r_anc] if not a.rapido else [1.0, r_anc]
    GRID_GL    = [1e-3, 3e-3, 1e-2, 3e-2]     if not a.rapido else [3e-3, 1e-2]

    formas = [("gl+raw", "raw"), ("gl+lib", "lib"), ("gl+rownorm", "rownorm")]
    if vivo:
        formas.insert(2, ("gl+cos", "cos"))

    def poner_gl(v):
        E6.GL_LAMBDA = v
        globals()["GL_LAMBDA"] = v

    def evaluar(model, etiqueta, verbose=True):
        acc = eval_acc(model, te)
        pruned, ff, fp = run_pruning_pipeline(model, etiqueta)
        acc_p = eval_acc(pruned, te)
        red = 1.0 - fp / ff
        caida = 100 * (acc - acc_p)
        ok = caida < a.max_caida
        if verbose:
            print(f"    acc {acc:.4%} -> {acc_p:.4%} (caida {caida:+.2f} pp)  "
                  f"FLOPs -{red:.2%}  {'OK' if ok else 'ROTO'}")
        return dict(acc=acc, acc_podado=acc_p, caida_pp=caida, reduccion=red,
                    valido=bool(ok), flops_full=int(ff), flops_pruned=int(fp))

    def correr(brazo, forma, gl, ratio, semilla):
        poner_gl(gl)
        if ratio is not None:
            globals()["TARGET"] = ratio
        m = train_arm(tr, forma, True, ep, semilla)
        return evaluar(m, f"{brazo}@s{semilla}", verbose=False)

    # ================================================================== AJUSTE
    configs = None
    if a.modo in ("tune", "todo"):
        print("\n" + "=" * 74)
        print(f"AJUSTE: rejilla sobre semillas FUERA DE MUESTRA {TUNE_SEEDS}")
        print("=" * 74)
        cand = []
        for g in GRID_GL:
            cand.append(("gl", None, g, None))
        for nombre, forma in formas:
            for rt in GRID_RATIO:
                cand.append((nombre, forma, GL0, rt))

        medidas = []
        for brazo, forma, gl, ratio in cand:
            cfg = f"GL={gl:.0e}" + ("" if ratio is None else f" ratio={ratio:.3g}")
            rs = [correr(brazo, forma, gl, ratio, sd) for sd in TUNE_SEEDS]
            todas_ok = all(r["valido"] for r in rs)
            red = sum(r["reduccion"] for r in rs) / len(rs)
            acc = sum(r["acc"] for r in rs) / len(rs)
            cai = max(r["caida_pp"] for r in rs)
            print(f"  {brazo:12} {cfg:22} FLOPs -{red:.2%}  acc {acc:.4%}  "
                  f"peor caida {cai:+.2f} pp  {'OK' if todas_ok else 'ROTO'}")
            medidas.append(dict(brazo=brazo, forma=forma, gl=gl, ratio=ratio,
                                reduccion=red, acc=acc, peor_caida=cai,
                                valido=todas_ok))

        configs = {}
        print("\n  CONFIGURACION ELEGIDA POR BRAZO (mayor FLOPs con las dos semillas OK):")
        for br in ["gl"] + [n for n, _ in formas]:
            val = [m for m in medidas if m["brazo"] == br and m["valido"]]
            if not val:
                print(f"    {br:12} SIN CONFIGURACION VALIDA")
                configs[br] = None
                continue
            best = max(val, key=lambda m: m["reduccion"])
            configs[br] = dict(forma=best["forma"], gl=best["gl"], ratio=best["ratio"])
            cfg = f"GL={best['gl']:.0e}" + ("" if best["ratio"] is None
                                            else f" ratio={best['ratio']:.3g}")
            print(f"    {br:12} {cfg:22} (FLOPs -{best['reduccion']:.2%} en ajuste)")
        json.dump(dict(configs=configs, medidas=medidas, tune_seeds=TUNE_SEEDS),
                  open(a.configs, "w"), indent=2)
        print(f"\n  guardado en {a.configs}")

    # ============================================================== EVALUACION
    res = []
    if a.modo in ("eval", "todo"):
        if configs is None:
            configs = json.load(open(a.configs))["configs"]
        print("\n" + "=" * 74)
        print(f"EVALUACION: configuraciones fijas sobre las {len(EVAL_SEEDS)} semillas")
        print("=" * 74)
        for br, cfg in configs.items():
            if cfg is None:
                print(f"\n  {br}: sin configuracion valida, se omite")
                continue
            print(f"\n  {br}  (GL={cfg['gl']:.0e}"
                  + ("" if cfg["ratio"] is None else f" ratio={cfg['ratio']:.3g}") + ")")
            rs = [correr(br, cfg["forma"], cfg["gl"], cfg["ratio"], sd)
                  for sd in EVAL_SEEDS]
            for sd, r in zip(EVAL_SEEDS, rs):
                print(f"    s{sd:<5} acc {r['acc']:.4%} -> {r['acc_podado']:.4%}"
                      f"  FLOPs -{r['reduccion']:.2%}  {'OK' if r['valido'] else 'ROTO'}")
            red = [r["reduccion"] for r in rs]
            res.append(dict(brazo=br, config=cfg, por_semilla=rs,
                            media=sum(red) / len(red),
                            n_validas=sum(1 for r in rs if r["valido"]),
                            acc_media=sum(r["acc"] for r in rs) / len(rs),
                            accp_media=sum(r["acc_podado"] for r in rs) / len(rs)))

        # controles sin hiperparametro propio
        print("\n  none (sin nada)")
        rs = [evaluar(train_arm(tr, None, False, ep, sd), "none", False)
              for sd in EVAL_SEEDS]
        res.append(dict(brazo="none", config=None, por_semilla=rs,
                        media=sum(r["reduccion"] for r in rs) / len(rs),
                        n_validas=sum(1 for r in rs if r["valido"]),
                        acc_media=sum(r["acc"] for r in rs) / len(rs),
                        accp_media=sum(r["acc_podado"] for r in rs) / len(rs)))

        print("\n" + "=" * 74)
        print(f"RESUMEN SOBRE {len(EVAL_SEEDS)} SEMILLAS")
        print("=" * 74)
        for r in sorted(res, key=lambda r: -r["media"]):
            print(f"  {r['brazo']:12} FLOPs -{r['media']:.2%}   "
                  f"acc {r['acc_media']:.4%} -> {r['accp_media']:.4%}   "
                  f"validas {r['n_validas']}/{len(EVAL_SEEDS)}")
        print(f"  {'ancla exp6':12} FLOPs -{red_anc:.2%}   "
              f"acc {acc_anc:.4%} -> {acc_anc_p:.4%}   (1 semilla, lambda fijo)")

        base = next((r for r in res if r["brazo"] == "gl"), None)
        if base:
            print(f"\n  APORTE SOBRE GROUP LASSO SOLO, pareado por semilla:")
            for r in res:
                if not r["brazo"].startswith("gl+"):
                    continue
                d = [x["reduccion"] - y["reduccion"]
                     for x, y in zip(r["por_semilla"], base["por_semilla"])]
                gana = sum(1 for v in d if v > 0)
                print(f"    {r['brazo']:12} {100*sum(d)/len(d):+6.2f} pp de media,"
                      f"  gana {gana}/{len(d)} semillas,"
                      f"  accuracy {100*(r['acc_media']-base['acc_media']):+.2f} pp")
            print("\n  Leer el aporte de FLOPs JUNTO a la diferencia de accuracy:")
            print("  mas FLOPs con menos accuracy no es una mejora, es un canje.")

    json.dump(dict(ancla=dict(acc=acc_anc, acc_podado=acc_anc_p,
                              caida_pp=caida_anc, reduccion=red_anc,
                              reproduce=bool(reproduce),
                              ratio_efectivo=r_anc, lambda_fijo=LAM_FIJO),
                   cribado=filas_cribado, coseno_vivo=vivo, brazos=res, configs=configs,
                   config=dict(modo=a.modo, tune_seeds=TUNE_SEEDS, eval_seeds=EVAL_SEEDS,
                               every_k=EVERY_K, probes=PROBES,
                               gl_lambda=GL_LAMBDA, epocas=ep, semilla=SEED)),
              open(a.out, "w"), indent=2)
    print(f"\nGuardado en {a.out}")


if __name__ == "__main__":
    main()
