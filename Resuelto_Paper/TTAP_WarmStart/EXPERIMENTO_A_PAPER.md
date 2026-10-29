# Experimento A para reemplazar la sección 4.1

El notebook `20261028_Experimento_A_WarmStart.ipynb` ejecuta ahora el perfil
`TTAP-A-paper-five-budgets-v1`. Su modo inicial es `full` y el objetivo es
`multidimensional`. Antes de ejecutar todas las celdas, comprobar esos dos
parámetros y la carpeta persistente de resultados.

El perfil compara dos políticas constructivas y tres inicializaciones del
mismo MILP: cold, Benefit start y Density start. Cada una recibe presupuestos
independientes de 1, 5, 15, 60 y 600 segundos. Hay 30 semillas en cada tamaño
20/30/40/50/60: 150 instancias, 2250 optimizaciones y 300 greedy. Cold a
600 segundos ya está incluido; no existe otra fase de referencia.

La suma de presupuestos es 85.125 horas. No es una predicción del tiempo real
ni un límite garantizado de duración total. Cada búsqueda puede terminar antes
al certificar; generación, preparación, escritura y excesos del límite agregan
tiempo. Los modos `pilot` (20 tareas, semilla 1) y `screen` (20/40/60 y semillas
1/2/3) conservan los cinco presupuestos para comprobar el mismo protocolo.

## Ejecución y reanudación

1. Reiniciar el kernel después de sustituir el notebook.
2. Ejecutar la celda de carga. Descarga una revisión fijada y verifica hashes.
3. Revisar la configuración; usar una carpeta persistente y `RESUME=True`.
4. Ejecutar la celda de optimización y luego la de exportación.

La carpeta se llama `A_paper_<modo>_<huella>`. Las corridas anteriores de cuatro
presupuestos permanecen separadas. Para reanudar, conservar el mismo modo,
objetivo, pesos, referencias, fuentes y entorno. Se guardan checkpoints por job.
El archivo `batch_sessions.csv` registra el tiempo de cada sesión completada;
las columnas end-to-end por tratamiento cargan preparación compartida y no se
deben sumar como si fueran duración real de todo el lote.

El vector inicial contiene tareas, retornos y terminación. Se comprueba antes
de entregarlo a HiGHS y no se fijan sus variables. Una solución óptima de una
corrida larga nunca se utiliza como start de una corrida corta.

## Resultados para Overleaf

Después de una corrida completa, la subcarpeta `paper` contiene:

- `Table_1_Experiment_A_quality.tex`: los cinco paneles y muestras emparejadas.
- `Table_2_Experiment_A_runtime.tex`: segundos para MILP, milisegundos para greedy.
- `A1_solution_quality_coverage_aware.pdf`: beneficio frente a tareas, medias y SD.
- `A8_computational_effort.pdf`: esfuerzo por tamaño y relación arcos-tiempo.
- `Section_4_1.tex`: reemplazo breve de la sección con cifras de la corrida.
- `Section_3_3_impact_update.tex`: definición coherente con el objetivo elegido.
- CSV con observaciones, estadísticas, referencias certificadas y denominadores.

Los pilotos escriben en `paper_preview`, llevan una marca visible y no generan
`Section_4_1.tex`. La exportación completa exige las 2250 observaciones MILP y
los 300 greedy, sin duplicados. Un óptimo desconocido se conserva desconocido.

La Tabla 1 describe valor disponible en todas las instancias generadas. Si cold
no devuelve incumbent, se puntúa por separado el plan idle disponible como cero;
el incumbent original continúa ausente en los CSV. La factibilidad devuelta por
HiGHS, el respaldo externo y la certificación se reportan por separado. Las
distancias al óptimo se calculan por instancia antes de promediar y únicamente
sobre certificados del mismo modelo y objetivo.

Las tres curvas MILP de A1 se muestran juntas sólo cuando todas las variantes
certificaron todos los pares y sus beneficios coinciden. A8 muestra cada corrida
y distingue las inicializaciones; su eje vertical es tiempo de solver. Tabla 2
incluye construcción, conversión, carga y búsqueda, más los costos del start.

`A3` y `A9` a `A12` conservan el diseño por tamaño de tareas y sirven para la
sección 4.3: certificación, factibilidad devuelta, incumbent, distancia al óptimo
y aporte de inyectar el greedy frente a cold que conserva ese mismo respaldo.
Se exportan PDF y SVG vectoriales, PNG de 1200 dpi y una vista previa de 240 dpi.
El notebook muestra SVG. El PDF es el archivo que se usa en Overleaf. Líneas
y marcadores más finos reducen el peso visual sin interpolar ni cambiar datos.
El notebook `20261029_Reexportar_Experimento_A_Paper.ipynb` vuelve a generar
las figuras y tablas de una corrida completa sin ejecutar ninguna optimización.
Si falta la carpeta local, puede recuperar únicamente `metrics.csv` y el
manifiesto del resultado publicado, con verificación de integridad.

Las figuras A10/A11 muestran medias condicionadas a disponer de incumbent
(y, para brechas, de referencia certificada). Los marcadores abiertos y las
anotaciones indican muestras incompletas. A13 reúne factibilidad, beneficio y
brecha con los denominadores visibles. No interpretar un cero de brecha basado
en pocos incumbents como calidad en toda la cohorte. A12 expresa la ganancia
de inyección en puntos porcentuales del beneficio normalizado y conserva
valores negativos. El CSV `injection_win_tie_loss_counts.csv` añade victorias,
empates y pérdidas con tolerancia absoluta de 1e-6.

En Overleaf se requieren `booktabs`, `makecell` y `graphicx`. Subir las dos
figuras principales y los .tex juntos, o adaptar `graphicspath`. Revisar los
nombres de labels si el proyecto usa otros. El texto generado describe la
corrida; los autores deben revisar la interpretación antes del envío.

## Consistencia matemática

En `multidimensional`, pacientes, pasajeros no médicos y carga suministrada son
outputs distintos. No deben contarse personas dos veces ni atribuir beneficio
al peso corporal o a equipo que sólo apoya la operación. Los pesos son
compensatorios y no constituyen precedencia médica absoluta ni calibración
clínica. Las referencias 3/3/500 son escalas fijas del generador.

Actualizar la sección 3.3 y la fórmula de ambos greedy junto con la sección 4.1.
La introducción de la sección 4 debe hablar de dos políticas constructivas y
tres inicializaciones del mismo MILP. Recalcular B y Talcahuano con el mismo
objetivo antes de presentar todos los casos juntos. Sus notebooks y el protocolo
anterior se conservan. El perfil B de seis presupuestos se documenta por
separado en `EXPERIMENTO_B_PAPER.md`.

El programa guarda las nuevas corridas localmente y no publica datos,
resultados, gráficos ni manuscritos en GitHub.
