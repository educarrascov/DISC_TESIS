# Talcahuano: HiGHS, Greedy Benefit y Greedy Density

Abrir `Resuelto_Paper/20260829_TTAP_ambosMetodos_GPS_Impacto_Estandarizado.ipynb`,
reiniciar el kernel y ejecutar las celdas en orden. El escenario inicial mantiene
las siete áreas, seis helicópteros, treinta tareas, coordenadas y datos del
notebook original de la revisión `0be00802518836ed652035dad6d504cfbb614c8a`.

## Objetivo y políticas

El valor inicial es `OBJECTIVE_MODE='primary'`, igual que en la versión original.
La clase principal determina el resultado que genera beneficio; los demás
requerimientos consumen recursos sin añadir beneficio. `multidimensional` sólo
corresponde si esas cantidades son outputs de ayuda distintos y no superpuestos.
Los tres métodos usan el mismo objetivo. Elegirlo de forma consistente con A/B
y con la sección 3 antes de presentar los resultados juntos.

- Benefit selecciona el mayor beneficio inmediato factible.
- Density selecciona beneficio dividido por vuelo hacia la tarea más servicio,
  en minutos continuos. El retorno no está en este denominador, igual que en A/B.
- Ambos seleccionan el helicóptero disponible antes y reservan tiempo suficiente
  para retorno y recuperación. Los desempates usan beneficio, finalización y Task ID.
- Todas las rutas utilizadas terminan en Base; también cuando se completa la última tarea.
- HiGHS resuelve el mismo MILP de estado-arco, con límite inicial de 180 s y tolerancia 1e-6.

Las llamadas greedy construyen estados nuevos e independientes. Que produzcan
el mismo plan en una instancia no significa que sean la misma política. La
comparación muestra ambos resultados y señala curvas coincidentes.

## Salidas

La carpeta es `figures_ttap_GPS/<escenario>_<objetivo>_three_methods`. Contiene:

- `talcahuano_comparison.csv`: tres métodos, beneficio, tareas, tiempos y certificados.
- `talcahuano_metrics.csv`: métricas ampliadas y outputs contabilizados según el objetivo.
- `talcahuano_routes.csv`: rutas y consumos de los tres métodos.
- `talcahuano_helo_balance.csv`: seis helicópteros por método.
- `Table_Talcahuano_three_methods.tex`: tabla de una instancia, sin medias ni SD ficticias.
- Gantt separados para los tres métodos y un Gantt conjunto de tres paneles.
- Curvas acumuladas separadas y una comparación de tres métodos.
- `manifest.json`: objetivo, referencias, entorno, regla density y límite del solver.

Las figuras comparativas se guardan en PDF/SVG vectoriales y PNG de 2000 dpi.
Para Overleaf usar PDF. El nombre antiguo del PDF acumulado se conserva como
alias de la nueva comparación de tres métodos. La carpeta depende del objetivo
para no mezclar primary y multidimensional.

El eje de las curvas acumuladas es tiempo de misión en minutos. Los escalones
añaden beneficio al finalizar tareas; no representan la evolución computacional
anytime del solver. Los tres métodos comparten las escalas temporales y de beneficio.

Sólo se calcula brecha al óptimo cuando existe certificado. Un incumbent de
HiGHS sin certificado se identifica como tal. Si no hay incumbent, se conserva
como ausente, sin presentarlo como un plan de valor cero. Los tiempos del MILP
incluyen construcción; el manifiesto distingue el límite solicitado al solver.

## Cartografía y alcance del caso

`RUN_GIS_MAPS=False` permite ejecutar los métodos sin archivos GIS externos.
Las coordenadas y distancias GPS siguen activas. Para reproducir los mapas,
activar `RUN_GIS_MAPS=True` y disponer de los shapefiles/KMZ en las rutas `data/`
originales. Las celdas cartográficas conservan su contenido dentro de ese control.

La configuración de treinta tareas tiene propósitos médicos y de transporte de
personal; no contiene tareas de propósito exclusivamente cargo. En primary, la
carga de esas tareas es un requerimiento de apoyo, no otro beneficio. No se han
inventado tareas de carga para cambiar esa composición.

Al actualizar el paper, incluir Density en la tabla del caso, la leyenda de la
curva acumulada y el Gantt. No afirmar que Density mejora Benefit si sus
resultados coinciden. Esta comparación no añade warm starts a Talcahuano;
compara las dos políticas independientes con HiGHS sobre la misma instancia.
