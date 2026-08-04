# AUDIT : estado de los resultados (actualizado 2026-07-24)

> **NOTA DE ESTADO (2026-08-03).** Este documento es el registro de la auditoría
> del 24-jul y se conserva por su valor histórico. Dos de sus contenidos están
> **superados**: (a) la reducción de FLOPs del 54.46% se re-midió el 31-jul con
> lambda optimizado y brazo de control, y el resultado es que group lasso solo
> alcanza la misma compresión, de modo que la penalización **no aporta
> compresión** y la sección de poda se retiró del preprint; (b) las cifras de
> retención de aquí son de la configuración anterior. Las vigentes son 0.173 ->
> 0.238 en capacidad absoluta y 84.1% de cociente de retención. Ver `CHANGELOG.md`
> y el preprint corto.

Este documento registra una auditoría del código y una re-medición de los
resultados publicados. Se mantiene por transparencia: algunos números
anteriores del README y del preprint han sido reemplazados.

## Hallazgo principal

En la familia de experimentos de LLM (LoRA sobre Llama-3-8B y GPT-2), la
penalización de Omega-S se calculaba sobre `p.data`, lo que la desacopla del
grafo de autograd. El término se sumaba a la pérdida como una constante y su
gradiente era cero: el regularizador no modificaba los pesos. Los números de
retención y de FLOPs en GPT-2 que dependían de esas corridas quedan invalidados
y han sido re-medidos.

La familia de experimentos de MLP (`exp1`, `exp3`-`exp6`) usa `layer.weight`,
está conectada correctamente y no se ve afectada.

## Qué se mantiene

- **Reducción de FLOPs en MLP, 54.46%** (`exp6`), penalización conectada y
  weight decay igualado en accuracy. Pendiente aislar la contribución marginal
  de Omega sobre group lasso.
- **Reducción de varianza de grados, 13×** (`exp1`), control aislado conectado.
- **Resultado teórico del FSRI** (`Tr(A^3) = Σλ_i^3`, contraejemplo cospectral):
  es matemático y no depende de ninguna corrida.

## Qué se re-midió

- **Retención en Llama-3-8B:** re-corrida con la penalización conectada, RNG
  dedicado por brazo. Se hicieron dos corridas: una de 5 semillas (omega
  calibrado) y una definitiva de 10 semillas. En la de 10 semillas, la forma
  log-ratio (`StochasticOmegaS`) deja mas capacidad de codigo que no
  regularizar en 9 de 10 semillas (HumanEval absoluto 0.168 -> 0.223, +33%
  relativo, p=0.011); como cociente de retencion la misma comparacion da 8 de
  10 (63.1% -> 76.6%), y sobre weight decay 9 de 10 y sobre EWC 8 de 10.
  La columna que antes se llamaba plasticidad era el numerador de la
  retencion, no una medida de la tarea nueva; ya no se llama asi. OJO: la seleccion de hiperparametros fue
  de DOS semillas, y las dos estan dentro de las diez evaluadas. Todos los
  brazos se barrieron: wd sobre {0, 0.01, 0.05, 0.1} -> 0.05; EWC sobre
  {100, 500, 1000, 5000, 10000} -> 1000 (el mas alto y el mas estable, desv
  0.011 frente a 0.06-0.08 de los demas), que coincide con el defecto; omega
  sobre {0.03, 0.1, 0.3, 0.5} -> 0.03, en el borde del rango. Ver seccion 4.5
  del paper. IMPORTANTE: el control de normas de fila (rownorm) es el UNICO
  brazo que NO se barrio. run_overnight.py solo barre wd, ewc, omega_raw y
  omega_lib; en la evaluacion rownorm se lanza con OMEGA_TARGET = optimo de
  omega_lib. Su colapso (0.207) es por tanto un artefacto de configuracion y
  la comparacion contra el no demuestra que Omega-S sea separable de igualar
  normas de fila. Es la pregunta abierta mas importante y el experimento mas
  barato que queda.

  RESUELTO el 26-jul-2026 con run_rownorm.py: el control se barrio sobre su
  propia rejilla {0.001, 0.003, 0.01, 0.03} en las mismas 2 semillas de ajuste
  y se evaluo en las 10. Optimo 0.001. Resultado: el control NO supera de forma
  fiable a no regularizar (6/10, p=0.38; 0.174 frente a 0.168 en capacidad
  absoluta) y tiene la mayor varianza de todos los brazos (desv 0.216); omega
  lo excede en 8/10 (p=0.055). Es evidencia de que el efecto no se reduce a
  equalizar normas de fila, sin cruzar la significacion convencional a diez
  semillas. CAVEAT: el 0.001 se eligio sobre las semillas 42 y 123, que son
  las dos peores de este brazo; no se reselecciono sobre las diez para no
  darle una ventaja de mejor-de-varios que omega no tuvo. Datos en
  results/rownorm_control_20260726.json. La forma cruda
  `Tr((WW^T)^3)` es el peor brazo (53.9%, por debajo de no regularizar). El
  repo se unifica sobre la forma log-ratio. Datos por semilla en
  results/merged_10seeds.json; código en experiments/rerun_retention.py y
  run_overnight.py.
- **Comparación vs EWC:** abierta. Con EWC sin ajustar el resultado es mixto
  (Omega adelante en 2/5). Un barrido de EWC con la misma vara está en curso.
- **FLOPs en GPT-2:** re-corrida conectada. Omega+group-lasso no supera a
  group-lasso solo (−0.72% vs −0.77%). La cifra anterior no sobrevive.


## MECANISMO MEDIDO (25-jul): opera vía varianza de grados, no clustering

Se midió directamente qué componentes del objetivo mueve la penalización durante
el entrenamiento (experiments/measure_structure.py, 10 semillas, brazos none y
omega_lib, solo tarea A). Resultado inequívoco y unilateral:

- El clustering C NO se mueve: dC ~= 0.000 en las 10 semillas (p.ej. semilla 456:
  C pasa de 0.999708 a 0.999702). C esta SATURADO en su techo (~0.9997) por el
  sigmoid que construye A = sigmoid(|W W^T|): comprime todo a [0.5,1), A queda
  casi todo-unos, y el clustering normalizado se clava cerca de 1 independiente
  de la estructura. El canal topologico esta INERTE por construccion.
- La varianza de grados (Coex) SI baja: 5-10% (semilla 456: 201.6 -> 182.8).
  Toda la accion esta ahi.
- Entre las 10 semillas, 0 estan dominadas por dC y 10 por dCoex -> NO hay
  mecanismo doble ni segundo regimen. El patron bimodal de la retencion no se
  explica por dos mecanismos estructurales en estos datos.

INTERPRETACION HONESTA: Omega-S es topologico POR DEFINICION MATEMATICA (su
objetivo es Tr(A^3)), pero TAL COMO ESTA FORMULADO y en estas condiciones opera
reduciendo la varianza de grados, correlacionada r~0.60 con la varianza de
normas de fila (queda ~40% sin explicar por normas). NO es, en la practica, un
metodo topologico basado en clustering. Esto revisa la EXPLICACION, no el
resultado: la ventaja de retencion (9/10 vs wd, 8/10 vs EWC, mejor plasticidad,
sin datos de la tarea anterior) SIGUE EN PIE.


