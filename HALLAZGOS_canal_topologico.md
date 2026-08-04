> **AVISO (31-jul-2026). ESTE DOCUMENTO ESTA SUPERADO.**
>
> Lo que aqui se plantea como linea abierta y experimento pendiente **se
> corrio**, y el resultado es **negativo**: la construccion coseno hace lo que
> se esperaba de ella (el exceso C/D pasa de 1.0000 a 1.36 de mediana y el
> signo de la penalizacion controla la direccion durante el entrenamiento),
> y **empeora la retencion en las diez semillas** (Wilcoxon W=0, p=0.002).
>
> Ademas se midio que **tres de los cuatro factores del objetivo estan
> numericamente inertes**, no solo el clustering: elasticidades 0.0000, 0.0000
> y 0.0001 frente a 0.0091 de la varianza de grados.
>
> Todo ello esta en el preprint (seccion *Testing the contrast-preserving
> reformulation*), en `CHANGELOG.md` y en
> `results/cosine_construction_20260731.json`. Este fichero se conserva como
> registro del razonamiento que llevo al experimento, no como estado actual.

---

# Restaurar el canal topológico de Omega-S: hallazgos y siguientes pasos

**Estado:** línea de trabajo abierta, NO incluida en el preprint. Documento
de trabajo interno para retomar más adelante.
**Fecha:** 25-jul-2026.

## Contexto

La medición directa del mecanismo (ver `experiments/measure_structure.py` y
la sección "What the penalty actually moves" del preprint) mostró que el
término de clustering `C` de Omega-S está **inerte**: `A = sigmoid(|W·Wᵀ|)`
comprime todas las afinidades a `[0.5, 1)`, la matriz queda casi todo-unos, y
el clustering normalizado `Tr(A³)/‖A‖³` se clava en ~1.0 con independencia de
la estructura. La penalización solo puede mover la varianza de grados (Coex).

La pregunta de esta línea: **¿se puede reformular la construcción de `A` para
que `C` vuelva a ser informativo, y aporta algo a la retención un término de
clustering así reformulado?** Contestarla cerraría a la vez (a) la pregunta de
si el método puede ser genuinamente topológico y (b) cuánto del ~40% de la
varianza de grados no explicado por normas de fila es estructura real.

## Lo que se probó (test de saturación, `experiments/reform_c.py`)

Se midió `C` sobre los pesos LoRA efectivos de Llama-3-8B (semilla 42, tras la
tarea A), con la construcción original y tres reformulaciones. Criterio de
"descomprime": `C_mean < 0.95` con spread entre módulos `> 0.01`.

| Vía | Construcción de A | C_mean | spread | ¿descomprime? |
|---|---|---|---|---|
| Original | `sigmoid(\|W·Wᵀ\|)` | 1.0000 | 0.0000 | No (saturado) |
| Temperatura | `sigmoid(\|W·Wᵀ\|/τ − b)`, b=mediana | 0.99 (τ=0.1) | 0.064 | No |
| Grado | `D^{-1/2} A D^{-1/2}` | 0.9997 | 0.0035 | **No** |
| Rango | CDF empírica de `\|W·Wᵀ\|` → [0,1] | **0.693** | **0.179** | **Sí** |

## Hallazgos

1. **La saturación es de escala, no de grado.** La normalización por grado
   (`D^{-1/2} A D^{-1/2}`), que era la vía con respaldo bibliográfico
   (Filan et al. 2021, "Clusterability in Neural Networks", regularizan el
   laplaciano normalizado; y arxiv 0704.0686 sobre clustering ponderado), **no
   descomprime `C`**. Razón: normalizar por grado una matriz que el sigmoid ya
   aplastó a casi-unos no restaura contraste. El problema está aguas arriba, en
   el sigmoid sobre un `|W·Wᵀ|` de rango enorme (valores ~1e3-1e14), no en la
   normalización.

2. **La temperatura tampoco basta.** Dividir por τ no vence el rango gigante de
   `|W·Wᵀ|`; con cualquier τ razonable el sigmoid satura igual. Solo τ=0.1 roza
   un contraste mínimo (0.064), insuficiente.

3. **Solo el mapeo por rango descomprime**, y con holgura (C 1.0 → 0.69, spread
   0.18). Funciona precisamente porque **descarta las magnitudes por completo**
   y deja `C` dependiente solo del orden de las afinidades. Es coherente con que
   el problema fuera el rango de magnitudes, pero es la transformación más
   agresiva y la de menor respaldo teórico previo.

## Lo que queda por hacer (fase 2, no ejecutada)

Descomprimir `C` es necesario pero **no suficiente**. Falta la pregunta que
decide todo: **¿penalizar el `C` reformulado por rango mejora la retención?**

Experimento diseñado, pendiente de correr (~20 celdas, 4 brazos × 5 semillas,
~2-3 h de pod, ~10-12 USD):

- `clustering_only`: penaliza solo el `C` reformulado por rango.
- `degvar_only`: penaliza solo la varianza de grados (el control de normas
  bien hecho, re-calibrado para no autodestruirse).
- `omega_lib` y `none` como referencias.

Lecturas:
- Si `clustering_only` aporta retención sobre `degvar_only` → hay una variante
  topológica real; el canal de clustering, una vez descomprimido, contribuye.
- Si no aporta → el mecanismo útil es la varianza de grados y punto; cierra la
  cuestión del 40% en negativo, también publicable.

## Riesgos técnicos anotados (para quien lo retome)

- **El gradiente del rango puede no ser útil.** El mapeo por rango
  (`argsort`/`searchsorted`) es casi constante a trozos, así que su gradiente
  puede ser ruidoso o casi nulo. Antes de lanzar la corrida hay que verificar en
  un smoke que la norma del gradiente del brazo `clustering_only` es no trivial
  (como se hizo con `assert_connected`). Si es minúscula, hará falta una versión
  **suave** del rango (p. ej. soft-rank diferenciable) en vez del rank duro.
- **Coste de memoria del rango.** El `argsort` directo sobre matrices 4096×4096
  (~16M elementos por módulo) agota la RAM; la implementación actual estima la
  CDF con `searchsorted` contra una submuestra de 200k, que es viable. Mantener
  ese muestreo.
- **Encuadre honesto para el paper si se persigue:** la reformulación que
  funciona (rango) NO es la que la bibliografía sugería (grado). Habría que
  decir explícitamente que la normalización estándar por grado no bastó y que
  solo descartar magnitudes por completo (rango) restaura el contraste. Es un
  hallazgo en sí, no un detalle a esconder.

## Ficheros

- `experiments/reform_c.py` : test de saturación con las tres vías.
- `results_reform/sat_test.json` : resultados del test (semilla 42).
- `experiments/measure_structure.py`, `analyze_structure.py` : la medición de
  mecanismo que originó esta línea.
