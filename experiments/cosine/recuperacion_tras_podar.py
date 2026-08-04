# =============================================================================
# ¿SOBREVIVE LA VENTAJA DE FIABILIDAD AL REAJUSTE TRAS PODAR?
#
# El criterio de validez que usamos hasta ahora exige que la poda cueste menos
# de 1 pp de accuracy SIN recuperacion ninguna. Pero cualquier tuberia de
# compresion real poda y DESPUES reentrena unas epocas. Es practica estandar.
#
# Si un reajuste corto rescata las semillas fallidas de group lasso solo,
# entonces lo que mediamos no era "no se puede podar" sino "no se puede podar
# sin reentrenar", y la ventaja de fiabilidad de la forma cruda se encoge o
# desaparece. Mejor saberlo antes de afirmar nada.
#
# Coste: los brazos de configs.json x 10 semillas x (5 epocas + FT_EPOCHS).
# Con --solo-clave se limita a gl y gl+raw, que son los dos que deciden.
# =============================================================================
import json
import os
import sys
import glob

# Colab: el import usa el sys.path del kernel, que no incluye el directorio de
# trabajo. Los !python funcionan porque lanzan un proceso nuevo; el import no.
_c = (glob.glob("/content/**/exp6_wd_vs_omega_pruning.py", recursive=True)
      + glob.glob("exp6_wd_vs_omega_pruning.py"))
assert _c, ("no encuentro exp6_wd_vs_omega_pruning.py: vuelve a ejecutar las "
            "dos celdas de %%writefile del notebook")
_ruta = os.path.dirname(os.path.abspath(_c[0]))
if _ruta not in sys.path:
    sys.path.insert(0, _ruta)
os.chdir(_ruta)
print("modulos en:", _ruta)

import torch
import torch.nn.functional as F

from exp6_wd_vs_omega_pruning import (
    DEVICE, LR, get_loader, build_model, eval_acc, run_pruning_pipeline,
    get_linear_layers, group_lasso_penalty,
)
import exp6_wd_vs_omega_pruning as E6
import exp6b_flops_by_form as X

FT_EPOCHS   = 2      # epocas de reajuste tras podar
MAX_CAIDA   = 1.0    # mismo criterio de validez que antes
SOLO_CLAVE  = True   # True = solo gl y gl+raw (20 corridas). False = todos.
EVAL_SEEDS  = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

# Configuraciones elegidas en el ajuste sobre semillas fuera de muestra
# (7077, 8088). Van a mano para no depender de configs.json, que se pierde si
# Colab reinicia la sesion.
CONFIGS = {
    "gl":         {"forma": None,  "gl": 1e-2, "ratio": None},
    "gl+raw":     {"forma": "raw", "gl": 1e-2, "ratio": 0.3},
    "gl+lib":     {"forma": "lib", "gl": 1e-2, "ratio": 0.3},
    "gl+cos":     {"forma": "cos", "gl": 1e-2, "ratio": 0.3},
    "gl+rownorm": {"forma": "rownorm", "gl": 1e-2, "ratio": 1.0},
}
configs = ({k: v for k, v in CONFIGS.items() if k in ("gl", "gl+raw")}
           if SOLO_CLAVE else CONFIGS)
tr, te = get_loader(train=True), get_loader(train=False)
print(f"brazos: {list(configs)}   semillas: {len(EVAL_SEEDS)}   "
      f"reajuste: {FT_EPOCHS} epocas\n")


def reajustar(model, epochs=FT_EPOCHS):
    """Reentrenamiento estandar tras podar: solo la tarea, sin penalizaciones.
    Es lo que hace cualquier tuberia de compresion en produccion."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for x, y in tr:
            opt.zero_grad()
            F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
    return model


filas = []
for brazo, cfg in configs.items():
    if cfg is None:
        continue
    print("=" * 74)
    print(f"BRAZO {brazo}   GL={cfg['gl']:.0e}"
          + ("" if cfg["ratio"] is None else f"  ratio={cfg['ratio']:.3g}"))
    print("=" * 74)
    for sd in EVAL_SEEDS:
        E6.GL_LAMBDA = cfg["gl"]
        X.GL_LAMBDA = cfg["gl"]
        if cfg["ratio"] is not None:
            X.TARGET = cfg["ratio"]
        m = X.train_arm(tr, cfg["forma"], True, X.EPOCHS, sd)
        acc = eval_acc(m, te)
        pruned, ff, fp = run_pruning_pipeline(m, f"{brazo}@s{sd}")
        acc_p = eval_acc(pruned, te)
        pruned = reajustar(pruned)
        acc_ft = eval_acc(pruned, te)
        red = 1.0 - fp / ff
        c_sin, c_con = 100 * (acc - acc_p), 100 * (acc - acc_ft)
        filas.append(dict(brazo=brazo, seed=sd, acc=acc, acc_podado=acc_p,
                          acc_reajustado=acc_ft, caida_sin=c_sin, caida_con=c_con,
                          reduccion=red, valido_sin=bool(c_sin < MAX_CAIDA),
                          valido_con=bool(c_con < MAX_CAIDA)))
        est = ("OK" if c_sin < MAX_CAIDA else
               ("RESCATADA" if c_con < MAX_CAIDA else "sigue rota"))
        print(f"  s{sd:<5} {acc:.4%} -> podado {acc_p:.4%} -> reajustado "
              f"{acc_ft:.4%}   caida {c_sin:+6.2f} -> {c_con:+6.2f} pp   {est}")

json.dump(filas, open("/content/recuperacion.json", "w"), indent=2)

print("\n" + "=" * 74)
print("FIABILIDAD, ANTES Y DESPUES DEL REAJUSTE")
print("=" * 74)
print(f"{'brazo':12} {'sin reajuste':>14} {'con reajuste':>14} {'rescatadas':>12}")
resumen = {}
for b in configs:
    f = [x for x in filas if x["brazo"] == b]
    if not f:
        continue
    sn = sum(x["valido_sin"] for x in f)
    cn = sum(x["valido_con"] for x in f)
    resumen[b] = (sn, cn)
    print(f"  {b:12} {sn:>10}/10 {cn:>13}/10 {cn-sn:>12}")

print("\n" + "=" * 74)
print("COMO LEERLO")
print("=" * 74)
if "gl" in resumen and "gl+raw" in resumen:
    g_sin, g_con = resumen["gl"]
    r_sin, r_con = resumen["gl+raw"]
    print(f"  ventaja SIN reajuste: {r_sin - g_sin:+d} semillas "
          f"({r_sin}/10 frente a {g_sin}/10)")
    print(f"  ventaja CON reajuste: {r_con - g_con:+d} semillas "
          f"({r_con}/10 frente a {g_con}/10)")
    print()
    if r_con - g_con >= r_sin - g_sin:
        print("  LA VENTAJA SOBREVIVE. El reajuste no rescata a group lasso solo,")
        print("  asi que la fiabilidad que aporta la forma cruda es real en una")
        print("  tuberia de compresion realista. Eso SI le importa a alguien que")
        print("  despliega modelos: menos reintentos de una corrida cara.")
    elif r_con - g_con > 0:
        print("  LA VENTAJA SE ENCOGE pero no desaparece. Hay que reportar las")
        print("  dos cifras y decir que parte de la diferencia la absorbe el")
        print("  reajuste estandar.")
    else:
        print("  LA VENTAJA DESAPARECE. Lo que mediamos no era 'no se puede podar'")
        print("  sino 'no se puede podar sin reentrenar', y eso lo resuelve una")
        print("  practica estandar. Hay que retirar la afirmacion de fiabilidad")
        print("  del paper, o reformularla como 'ahorra el reajuste posterior'.")
print("\nGuardado en /content/recuperacion.json")
