# Ejecutar los notebooks del paper

Los notebooks se encuentran en `Resuelto_Paper`. Sus perfiles actuales son:

| Notebook | Modo inicial | Presupuestos de cada tratamiento | Benchmark |
|---|---|---|---|
| `20261028_Experimento_A_WarmStart.ipynb` | full | 1/5/15/60/600 s | 600 s |
| `20261028_Experimento_B_WarmStart.ipynb` | pilot | 1/5/15/60/600/1800 s | 1800 s |
| `20261029_Reexportar_Experimento_A_Paper.ipynb` | sólo exportación | no optimiza | resultados terminados |

A conserva las 150 instancias de su perfil anterior de cinco presupuestos.
No necesita repetirse para cambiar las figuras. B usa un nuevo perfil de seis
presupuestos: 150 instancias, 2700 optimizaciones y 300 greedy en full.
Ambos comparan cold, Benefit start y Density start, con el mismo objetivo
multidimensional. Los presupuestos son búsquedas independientes y no existe
otra fase de referencia. Las configuraciones antiguas de cuatro presupuestos
siguen disponibles por `study_runner.configuration`, fuera de estos notebooks.

1. Descargar el notebook de la revisión publicada y reiniciar el kernel.
2. Ejecutar la carga con verificación de integridad.
3. Revisar `MODE`, objetivo y carpeta de salida antes de optimizar.
4. Para B, comprobar el piloto y luego cambiar `MODE='full'`.
5. Conservar `RESUME=True`, las fuentes y el entorno para reanudar.

Instrucciones y costos nominales:
[Experimento A](EXPERIMENTO_A_PAPER.md) y [Experimento B](EXPERIMENTO_B_PAPER.md).
Los límites son tiempos de solver, no plazos de respuesta de extremo a extremo.

Para Overleaf usar PDF vectorial. El notebook muestra SVG vectorial y también
exporta PNG de 1200 dpi. La reexportación no modifica datos ni ejecuta HiGHS.
El perfil A conserva su huella de configuración; cambiar sólo el estilo de
las figuras no invalida los resultados ya calculados.

Las trazas permiten la figura A14/B14 de reloj observado dentro de una corrida
larga. Los otros gráficos por presupuesto usan límites nominales de HiGHS,
que pueden excederse. No confundir ambas mediciones.

Las corridas se guardan localmente. El código no publica resultados ni manuscritos.

## Si apareció «Notebook and runtime differ»

Descargar de nuevo ambos notebooks desde la rama publicada y reiniciar el
kernel antes de ejecutar las celdas. La revisión inicial contenía tres hashes
calculados con una línea en blanco adicional al final de los módulos frozen.
La versión corregida verifica los bytes publicados y usa una carpeta aislada.
También acepta CRLF al copiar desde un checkout de Windows. Una versión local
distinta se conserva y se utiliza la revisión fijada del repositorio.

No eliminar la verificación ni editar los archivos frozen para hacerla pasar.
Si se corta una descarga, volver a ejecutar la primera celda; se conservan los
archivos válidos y se reintentan los que faltan. Después de cargar módulos de
otra carpeta, reiniciar el kernel antes de continuar. El modelo y los
parámetros experimentales no cambian con esta corrección.
