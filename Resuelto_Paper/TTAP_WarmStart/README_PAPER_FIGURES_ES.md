# Figuras del paper: A/B terminados, sin repetir HiGHS

Este complemento sólo lee `manifest.json`, `metrics.csv` y `trace.csv`. No importa
HiGHS, no modifica el objetivo y no cambia los módulos verificados de los
notebooks que ya están ejecutados. Los resultados publicados se fijan al commit
`2cd8035c8417b3c4f5107c57ee174beef046ced3`.

## Para los kernels actuales

Agrega al **final** de cada notebook la celda del archivo
`CELDA_REGENERAR_FIGURAS.py`, publicado junto al notebook de comparación.
Usa el `RUN_DIR` que ya existe. Después ejecuta la llamada indicada para A o B.
No ejecutes otra vez `run_paper_experiment_a/b`, la configuración, las instalaciones
ni las celdas que verifican el runtime. Tampoco es necesario reiniciar el kernel.

El resultado se guarda en `RUN_DIR / "paper_restyled"`, por separado de `paper`.
Las figuras se cierran después de guardarlas; se muestran mediante las imágenes
de previsualización para no acumular grandes lienzos en memoria.

## Notebook independiente

Abre `20261107_Comparacion_A_B_Figuras_Paper.ipynb`. Puede usar las dos carpetas
locales de resultados o descargar los seis archivos publicados. No requiere
token de GitHub ni una instalación de `highspy`. Las dependencias son `numpy`,
`pandas`, `matplotlib` e `IPython`, habituales en los notebooks existentes.
Si falta una, instálala en el entorno de visualización que estés usando; no
cambies el entorno de una optimización aún en ejecución.

La celda de estilo concentra colores, fuentes, tamaños y DPI. Los gráficos
se exportan en PDF y SVG vectoriales, PNG y una previsualización de menor tamaño.
El estándar visual recupera los notebooks ANYTIME: ejes negros, cuadrícula
discontinua, lienzos amplios, líneas de 1.8--2 puntos y bandas de dispersión.
No se interpolan ni se suavizan los resultados para inventar valores intermedios.

`png_dpi=600` es el valor inicial. Para un PNG como los antiguos puedes usar
`png_dpi=2000`; la geometría y los datos no cambian. Esa exportación requiere
mucha más memoria, especialmente en figuras de varios paneles. El PDF vectorial
es la opción para Overleaf y permanece nítido con cualquiera de esos DPI.
`STYLE["available_ylim"] = (0, 0.65)` permite ampliar verticalmente la figura
conjunta A14/B14 manteniendo la misma escala para los dos experimentos.

## Qué muestra cada figura

| Archivo | Contenido y muestra |
| --- | --- |
| `A1/B1_solution_quality_certified_matched` | Sólo instancias certificadas; ambos greedy se evalúan en exactamente esas mismas instancias. Media y SD muestral. Puntos huecos cuando la muestra es condicional. |
| `A2/B2_end_to_end_computational_time` | Mediana e intervalo intercuartílico; todos los casos, incluidas las búsquedas sin certificado. |
| `A8/B8_computational_effort_Cold` | Estética original del ANYTIME: dispersión por tareas y tamaño del MILP, mediana e IQR. Todos los casos; no se ocultan las corridas difíciles. |
| `A3/B3`, `A9/B9`, `A10/B10`, `A11/B11` | Certificación, incumbents, media condicional del incumbent y brecha condicional. Son corridas independientes según límite configurado; los denominadores están en `budget_by_task.csv`. |
| `A12/B12_warm_start_attribution` | Ganancia sobre el greedy inicial y ganancia sobre cold con ese greedy de respaldo; comparaciones pareadas por límite configurado. |
| `A13/B13_budget_quality_overview` | Disponibilidad, beneficio disponible sobre los 150 casos y brecha del plan disponible frente a óptimos conocidos. |
| `A14/B14_observed_anytime_quality` | Disponibilidad observada dentro de una única búsqueda larga, sin incorporar información posterior al punto temporal. |
| `AB14_observed_available_benefit` | Panel central de A14 a la izquierda y de B14 a la derecha. Todos los casos; no se filtra por certificación futura. |
| `AB5_observed_certification_Cold` | Recomendación para hablar de presupuesto de tiempo efectivamente observado. Cinco paneles por cantidad de tareas; umbrales de 80 % y 95 %. |
| `AB5_budget_certification_Cold` | Misma presentación, con resultados finales de las corridas independientes por límite configurado. No son plazos estrictos de pared. |
| `AB5_*_GBStart/GDStart` | Fronteras adicionales para las dos inicializaciones; apropiadas para comparar o llevar al suplemento. |
| `AB_budget_quality_dashboard` | Resumen A/B de resultados por límite configurado; denominadores iguales para beneficio disponible, distintos para brechas conocidas. |
| `AB15_paired_injection_effect` | Incremento adicional de inyectar el inicio frente a guardar el mismo greedy como respaldo externo. No confundirlo con la mejora respecto de cold solo. |

`quality_cohort="Cold"` reproduce la selección del paper original. También
se admiten `"common"` (certificadas por las tres inicializaciones) y `"reference"`
(óptimo certificado por alguna corrida del mismo modelo). En estos datos A/B,
Cold y common seleccionan los mismos casos. Se dibuja una sola curva certificada
porque las tres inicializaciones alcanzan los mismos valores en esa muestra.

## Tiempo configurado y tiempo observado

En B hay sobrepasos importantes: por ejemplo, `B6x7_stress_T60_S30` registra
10.327,53 s de solve wall time en la corrida cold con límite de 600 s; el
tiempo interno registrado por HiGHS también es aproximadamente 10.327,50 s.
Por eso no se deben presentar los resultados finales de ese job como si
hubiesen estado disponibles antes de los 600 s.

Este complemento conserva las dos lecturas y las etiqueta por separado:

- **Límite configurado:** resultados al terminar cada job. Sirve para reproducir
  las tablas previas, con su salvedad temporal.
- **Tiempo observado:** en A usa la búsqueda de 600 s y en B la de 1800 s.
  Sólo cuenta callbacks con tiempo menor o igual al punto considerado. Un
  certificado se cuenta después de finalizar la búsqueda con estado óptimo.

La segunda lectura no incluye generación de red, preparación del modelo ni
construcción/validación del greedy. Por tanto, tampoco es un plazo operativo
desde la recepción de una misión. El registro puede ser conservador: un
callback no identifica necesariamente el primer instante interno de hallazgo.

## Tablas para Overleaf

- En las carpetas A/B: tablas 1--4 con los mismos cinco campos y paneles del
  formato existente. Calidad usa la muestra certificada; runtime usa todos.
- `Table_AB_observed_budget.tex`: tabla recomendada para acompañar A14/B14;
  incluye certificados, factibles observados sin certificar, ausencia de
  incumbent observado, media del incumbent, beneficio disponible y brecha.
- `Table_AB_budget_status.tex`: las mismas categorías para los jobs finalizados
  por límite configurado. No confundirla con los checkpoints observados.
- `Table_AB_certification_thresholds.tex`: primer límite probado que alcanza
  80 % o 95 %; un guion significa que el objetivo no se alcanzó en los límites
  ensayados. No se extrapola.
- `Figures_AB_overleaf.tex`: bloques de figuras y captions en inglés. Copia sólo
  los bloques de las figuras elegidas, y conserva la distinción temporal.

Se requieren `booktabs`, `makecell` y `graphicx`.

## Qué incorporaría al artículo

En 4.1 y 4.2 sustituiría las figuras principales de calidad por A1/B1 y
sus respectivas tablas corregidas. Mantendría los puntos condicionales de B
con el número certificado visible. No añadiría tres curvas idénticas de óptimos.

En la sección de presupuesto usaría `AB14_observed_available_benefit` y la
tabla de checkpoints observados. Si se conserva la figura 5, usaría su versión
observada con los umbrales. El dashboard, las tres fronteras por inicialización
y el análisis de atribución pueden quedar en el suplemento; no es necesario
poner todo en el cuerpo del paper.

La mejora de disponibilidad no basta para afirmar que el warm start acelera
la optimización. La ganancia adicional de inyectar frente a cold + el mismo
greedy es pequeña en estos resultados. Conviene mostrarlo con claridad,
sin buscar una conclusión favorable que los datos no sostienen.

## Archivos de auditoría

`publication_manifest.json` identifica el objetivo y las convenciones de las
figuras. `AB_measured_time_audit.csv` conserva los sobrepasos, sin recortar los
tiempos registrados. Los CSV de calidad, presupuesto, callbacks y atribución
permiten revisar los denominadores y reproducir los valores de las tablas.

Los encabezados de los notebooks warm start originales todavía mencionan
"multidimensional" en ciertos textos, pero las dos corridas publicadas usan
`primary`. Corrige esa descripción editorial; no cambies la función objetivo
para regenerar estas figuras ni combines estas corridas con otras de distinto
objetivo.
