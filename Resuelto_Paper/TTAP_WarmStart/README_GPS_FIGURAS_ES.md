# Figuras del GPS ya ejecutado

`gps_publication_figures.py` genera figuras a partir de `METHOD_RESULTS` y
`df_metrics` del notebook GPS de agosto. También puede leer los CSV ya
exportados. No llama a los greedy, al generador ni a HiGHS.

Se revisaron los dos notebooks en el commit
`0d2c3476da6f7d19fb309b8cfb50e59b9ed648e5`. El notebook de julio se usa como
referencia visual: los valores, rutas y métricas siempre proceden de la corrida
actual. Se conservan las tres alternativas: Greedy Benefit, Greedy Density y
HiGHS, con su estado de certificación real.

## Agregar una celda al final

Si el módulo está en `TTAP_WarmStart`, junto al notebook dentro de
`Resuelto_Paper`, puedes usar:

```python
from TTAP_WarmStart import gps_publication_figures as gps_figures

GPS_STYLE = gps_figures.gps_style(
    png_dpi=600,
    preview_dpi=160,
    formats=("pdf", "png", "svg"),
    curve_mode="events",
)

GPS_REPORT = gps_figures.export_gps_figures(globals(), style=GPS_STYLE)
```

Ejecuta esta celda después de la sección de métricas del notebook que ya
completaste. No hace falta reiniciar el kernel ni ejecutar las celdas de
optimización otra vez. Si sólo falta `df_metrics`, ejecuta únicamente su celda
de cálculo descriptivo.

## DPI, formatos y estilo

```python
GPS_STYLE["png_dpi"] = 1000      # O 600 / 2000.
GPS_STYLE["preview_dpi"] = 160   # Visualización en el notebook.
GPS_STYLE["formats"] = ("pdf", "png")

GPS_STYLE["colors"]["Greedy Benefit"] = "#ff7f0e"
GPS_STYLE["colors"]["Greedy Density"] = "#2ca02c"
GPS_STYLE["colors"]["HiGHS"] = "#1f77b4"
GPS_STYLE["figures"]["gantt"] = (14, 4.5)
GPS_STYLE["figures"]["cumulative"] = (10, 5)
GPS_STYLE["task_label_size"] = 10.5
```

Después llama otra vez a `export_gps_figures`. Sólo se redibujan las figuras.
Para ampliar los Gantt como en julio, puedes usar `("gantt": (18, 5))` en el
diccionario de tamaños; por ejemplo:

```python
GPS_STYLE["figures"]["gantt"] = (18, 5)
GPS_STYLE["figures"]["gantt_comparison"] = (18, 10.2)
GPS_REPORT = gps_figures.export_gps_figures(globals(), style=GPS_STYLE)
```

El PDF y el SVG conservan líneas y textos vectoriales. El PNG usa los DPI
seleccionados y las previsualizaciones usan su propio valor de DPI. Aumentar
el DPI no cambia el tamaño de las fuentes ni corrige por sí solo un lienzo
demasiado pequeño. Los PNG de Gantt amplios a 2000 DPI requieren bastante
memoria; el PDF vectorial permite mantener la nitidez sin ese coste raster.

## Curvas y Gantt

`curve_mode="events"` recupera las líneas con marcadores del notebook de julio.
Cada punto es el beneficio acumulado al terminar las tareas de ese instante;
se agrupan las finalizaciones simultáneas. Las líneas que los unen son guías
visuales. No se ajustan splines ni se inventan beneficios intermedios.

`curve_mode="step"` representa literalmente los saltos al completar las tareas.
Ambas opciones conservan exactamente los mismos puntos y el mismo beneficio
final. Los dos greedy tienen curvas coincidentes en la corrida publicada; se
usan líneas y marcadores diferentes para dejar visibles ambos métodos.

Los Gantt integran el margen de redondeo temporal en el tramo de vuelo dibujado.
La leyenda lo identifica como `Flight + rounding`. Por ejemplo, un vuelo de
2,1 min más 0,9 min de redondeo ocupa 3 min en la figura, seguido de 6 min de
servicio hasta el final registrado en 9 min. Las métricas de vuelo conservan
los 2,1 min originales. La integración es una convención gráfica y mantiene
los inicios, finales y duraciones de servicio y recuperación registrados.
Todos los métodos comparten límites de tiempo y los seis helicópteros,
incluidos los que no tengan actividades en otra configuración.

```python
GPS_STYLE["grid_allowance_mode"] = "flight"  # Integrar redondeo en el vuelo dibujado.
GPS_STYLE["show_base_labels"] = True
GPS_STYLE["xlim"] = (0, 125)             # Debe incluir todos los finales.
GPS_STYLE["benefit_ylim"] = (0, 0.70)    # Debe incluir el beneficio final.
```

Para recuperar la representación separada puedes usar
`GPS_STYLE["grid_allowance_mode"] = "separate"`. En ese modo,
`show_grid_allowance=True` dibuja el tramado y `False` deja el margen blanco.
Con el modo `flight` el parámetro `show_grid_allowance` no se utiliza.

## Archivos generados

La carpeta inicial es `Path(FIGURES_DIR) / "paper_restyled"`. En esta corrida:

```text
figures_ttap_GPS/
  GPS_7areas_6helos_30tasks_primary_three_methods/
    paper_restyled/
```

Para elegir otra carpeta, pasa `output_directory=Path("otra_carpeta")` a
`export_gps_figures`.

| Archivo base | Contenido |
| --- | --- |
| `GPS_gantt_greedy_benefit` | Gantt de Greedy Benefit. |
| `GPS_gantt_greedy_density` | Gantt de Greedy Density. |
| `GPS_gantt_highs` | Gantt de la solución disponible de HiGHS. |
| `GPS_gantt_comparison_three_methods` | Tres Gantt con la misma escala. |
| `GPS_cumulative_greedy_benefit` | Beneficio acumulado de Benefit. |
| `GPS_cumulative_greedy_density` | Beneficio acumulado de Density. |
| `GPS_cumulative_highs` | Beneficio acumulado de HiGHS. |
| `GPS_cumulative_comparison_three_methods` | Comparación de las tres curvas. |
| `GPS_metrics_final` | Seis paneles de métricas finales. |
| `GPS_metrics_table` | Cuadro resumido legible para revisión. |

Las figuras se guardan en los formatos seleccionados. Siempre se genera un
PNG ligero en `_preview` para mostrarlas dentro del notebook. Los formatos
de exportaciones anteriores permanecen si cambias la selección.

También se escriben copias de las métricas, coordenadas de los puntos
acumulados e intervalos dibujados. El CSV de intervalos conserva `Flight Time`
y añade `Displayed Flight Time` y `Displayed Service Start`, para distinguir
los datos originales de su representación. Estos archivos se guardan en la carpeta
nueva; los CSV originales permanecen como fuente de la corrida. El archivo
`GPS_figures_overleaf.tex` propone captions cuando se exporta PDF.

## Usar los CSV si el kernel ya está cerrado

```python
GPS_SOURCE = gps_figures.load_gps_csv(
    "figures_ttap_GPS/GPS_7areas_6helos_30tasks_primary_three_methods"
)
GPS_REPORT = gps_figures.export_gps_figures(GPS_SOURCE, style=GPS_STYLE)
```

La carpeta debe contener `manifest.json`, `talcahuano_routes.csv`,
`talcahuano_metrics.csv`, `talcahuano_comparison.csv` y
`talcahuano_helo_balance.csv`.

## Uso en el artículo

Para el cuerpo del paper elegiría la comparación acumulada y los Gantt
comparados, con una tabla breve de resultados. La figura de seis métricas
sirve para revisión o suplemento: algunas magnitudes aumentan cuando se
atienden más tareas, y no son objetivos que HiGHS minimice por separado.

Se trata de una instancia fija. No se agregan SD ni bandas de incertidumbre
como si hubiera repeticiones. Los valores finales, el estado de certificación
y los tiempos computacionales son los registrados en la corrida, no tiempos
medidos de nuevo al generar imágenes.
