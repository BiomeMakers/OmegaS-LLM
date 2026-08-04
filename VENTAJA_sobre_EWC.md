# La ventaja de Omega-S sobre EWC: qué se afirma y qué no

> **NOTA (31-jul-2026).** El argumento de este documento se mantiene, pero la
> comparacion primaria del preprint ya **no** es contra EWC. En el barrido de
> diez semillas tanto weight decay como EWC quedan **por debajo del brazo sin
> regularizador**, asi que batirlos dice menos de lo que parecia y el preprint
> los reporta como secundarios. La afirmacion que sostiene el paper es contra
> `none`: 9 de 10 semillas en capacidad absoluta, p=0.011. Lo que sigue intacto
> de este documento es el diferencial **operativo**, que es lo que de verdad
> distingue al metodo: no necesita datos de la tarea anterior, ni Fisher, ni una
> copia de los pesos.

---


**Fecha:** 25-jul-2026. Documento de registro. Incorporado al preprint v7.

## El argumento, en una frase

Omega-S le gana a EWC en el test (8 de 10 semillas) **y además no necesita
nada de lo que EWC requiere**: ni guardar una copia de los pesos de la tarea
anterior, ni calcular la matriz de Fisher, ni conservar acceso a los datos de
la tarea anterior.

## Por qué esto importa

EWC es el método **estándar** para el olvido catastrófico. Es el baseline
difícil, el que de verdad hay que batir (weight decay, en cambio, es el
estándar de la generalización, no del olvido: en este experimento juega fuera
de su terreno). Que Omega-S le gane a EWC en retención ya es el resultado
fuerte del paper.

Pero la comparación tiene una segunda capa que la hace más contundente: EWC
paga un coste operativo que Omega-S no paga.

Para construir su penalización, EWC debe:
1. Guardar una copia de los pesos óptimos de la tarea A.
2. Estimar una matriz de Fisher pasando **datos de la tarea A** por el modelo.

El punto 2 es el crítico: **EWC necesita conservar acceso a los datos de la
tarea anterior.** En producción eso a menudo no es posible (privacidad,
coste de almacenamiento, datos que caducan o que ya no se pueden usar).

Omega-S solo lee los pesos actuales del modelo. No necesita datos viejos, ni
Fisher, ni copia de pesos. Por tanto **le gana al estándar del olvido
mientras elimina el requisito operativo central de ese estándar.**

## Lo que NO se afirma (y es importante no afirmarlo)

**No se reclama que Omega-S sea más rápido ni consuma menos cómputo por
paso.** Esa afirmación sería incorrecta sin medirla:

- EWC calcula la Fisher una sola vez (al final de la tarea A) y luego su
  término por paso es barato.
- Omega-S calcula su penalización cada K pasos, e incluye un estimador de
  Hutchinson sobre `W·Wᵀ`, que tiene un coste no despreciable y que **no se ha
  benchmarkeado** contra el de la Fisher de EWC.

Así que en FLOPs por paso no está claro cuál es más barato. Lo que Omega-S sí
evita es el pico de coste y memoria de calcular y almacenar la Fisher y la
copia de pesos. Pero la ventaja que el paper reclama es **operativa** (no
depende de datos previos), no de velocidad bruta.

## Relación mecanística: Omega es primo de weight decay, no de EWC

Por mecanismo, Omega-S pertenece a la familia de los regularizadores **ciegos
a la tarea** (como weight decay): solo miran los pesos actuales. EWC es de otra
familia: usa información de la tarea anterior (la Fisher). Omega-S rinde mejor
que ambos, pero es mecánicamente pariente de weight decay, no de EWC. Esto es
coherente con el mecanismo medido (controla la varianza de grados del grafo de
pesos, sin usar datos de ninguna tarea).

## Cómo quedó en el preprint (v7)

- **Abstract, punto 1:** remata con "beats EWC while needing none of what EWC
  requires: no stored weights, no Fisher, no prior-task data".
- **Resultados:** un párrafo dedicado desarrolla el argumento completo (EWC es
  el estándar, necesita datos previos, Omega gana 8/10 sin ese requisito) y
  añade explícitamente la salvedad de que no se reclama menor cómputo por paso.
