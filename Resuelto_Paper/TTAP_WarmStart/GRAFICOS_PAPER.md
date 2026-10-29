# Corridas completas con el estilo gráfico del paper

Los notebooks A y B comienzan en `MODE='full'`, con `PNG_DPI=2000` y
`RESUME=True`. Guardan la corrida en `ttap_runs_paper_2000dpi`, una carpeta
separada de las corridas anteriores. Se puede ejecutar cada notebook desde la
primera celda hasta la última. El parámetro `MODE='pilot'` sigue disponible
para una comprobación corta.

El notebook original usaba PNG de 2000 dpi, PDF vectorial y vista de notebook
a 160 dpi. Esta revisión retoma los 2000 dpi y aplica el mismo estilo a A, B
y sus figuras por tiempo: tipografía, tamaños, bandas, marcadores y trazos.
Las vistas del notebook son SVG vectoriales.

## Parámetros de la celda de configuración

```python
MODE = 'full'
PNG_DPI = 2000
OUTPUT_ROOT = Path.cwd() / 'ttap_runs_paper_2000dpi'
RESUME = True
```

Para iniciar otra repetición completa, cambiar `OUTPUT_ROOT` por una carpeta
nueva antes de ejecutar las optimizaciones. Para continuar una corrida
interrumpida, mantener la carpeta, la configuración y el entorno originales.
`RESUME=False` no borra ni reemplaza una carpeta que ya contiene resultados.

## Exportación

- PNG: 2000 dpi, el mismo valor solicitado en el notebook original.
- PDF: formato vectorial para incorporar en Overleaf.
- SVG: formato vectorial para visualizar dentro del notebook.
- `_preview`: copias pequeñas de 240 dpi para una inspección rápida.

Los tamaños se fijan en pulgadas para figuras de una o dos columnas. A1/B1
son más horizontales; A8/B8 ajustan su proporción, leyenda y barra de color.
Los ejes por presupuesto muestran los tiempos evaluados, sin la acumulación
de marcas menores. Las bandas siguen representando SD o IQR según la figura.
Los cambios de trazo y anti-aliasing no interpolan ni modifican las observaciones.

La configuración gráfica común está en `paper_a_report.py`; B la reutiliza.
El módulo del reloj observado usa esa misma configuración. Los manifiestos
de exportación registran los dpi, los tamaños y la versión gráfica.

El original usaba lienzos mayores. Aquí se conserva un tamaño de impresión
legible en el paper y se evita crear PNG de tamaño innecesariamente grande.
Usar los PDF en Overleaf permite ampliar sin perder definición de líneas y texto.

## Tamaño del estudio

A full comprende 150 instancias, 2250 optimizaciones y 300 greedy. B full
comprende 150 instancias, 2700 optimizaciones y 300 greedy. Sus sumas de
presupuestos son 85.125 y 310.125 horas, respectivamente. Son sumas nominales,
no predicciones de duración: la certificación puede terminar antes y la
preparación o los excesos del límite pueden agregar tiempo.

Actualizar los notebooks y reiniciar el kernel antes de ejecutar la nueva
revisión. La carga verifica los archivos del runtime publicado. Las figuras
se generan después de resolver; su resolución no cambia el modelo ni el resultado.
