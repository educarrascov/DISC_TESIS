# Figura 5 y warm start: qué elegir y qué significa

La opción 5 conserva la lectura de la figura antigua: un panel por número de
tareas, con certificación de A y B frente al presupuesto. Debajo añade una
tabla con disponibilidad de incumbents, beneficio medio y diferencia respecto
del óptimo conocido. Para Overleaf, usaría la figura original y la tabla como
dos elementos separados: se leen mejor y permiten conservar la numeración.
La opción 1 genera precisamente esa figura separada; la opción 5 exporta la
tabla en LaTeX.

| Opción | Lo que permite ver | Límite |
| --- | --- | --- |
| 1 | La certificación original y tres figuras adicionales con el mismo diseño | Incluir las cuatro figuras ocuparía demasiado espacio en el paper |
| 2 | Los cuatro indicadores en paneles grandes | Agrupa los cinco tamaños de instancia |
| 3 | Disponibilidad, beneficio y diferencia para cada tamaño | Tiene más paneles; sirve mejor para revisar los resultados |
| 4 | Cuántas soluciones faltan, tienen valor cero o están certificadas; y su variabilidad | Es más apropiada para material complementario |
| 5 | La figura antigua más una tabla breve con los indicadores de Pedro | La tabla agrupa tamaños; los CSV conservan el detalle |

Las medias de incumbent excluyen ejecuciones sin solución. Las diferencias
se calculan por instancia y después se promedian, sólo cuando se conoce su
óptimo certificado. Hay 150 referencias en A y 117 en B. Esto importa: el 0%
de diferencia de A a 1 segundo corresponde a sólo dos incumbents entre 150
ejecuciones. Tampoco una media que baja con el presupuesto indica que las
soluciones individuales empeoraron: pueden haber entrado instancias más
difíciles en el conjunto de soluciones disponibles.

## Lo que hace el warm start

Primero construye una ruta factible con Greedy Benefit o Greedy Density.
Luego convierte esa ruta en los valores de las variables binarias de la red
estado-arco y comprueba las restricciones, la integridad y el beneficio.
Entrega ese vector a HiGHS mediante `setSolution`, antes de `run`.
HiGHS puede cambiar cualquier decisión: el greedy no queda impuesto.

Si el solver termina sin una solución mejor, todavía disponemos del plan
greedy validado. Eso es útil operacionalmente, pero no demuestra que la
inyección mejoró la búsqueda. Por eso hay dos comparaciones distintas:

* **Mejora respecto del greedy:** valor final del warm start menos valor inicial.
* **Aporte de inyectarlo:** valor final del warm start menos el mejor entre el
  nuevo cold y ese mismo greedy conservado fuera del solver.

La segunda evita atribuir al solver un beneficio que ya teníamos. Puede ser
negativa si el cold encuentra algo mejor. Los archivos de resultados
conservan ese signo. Ambas comparaciones usan la misma instancia y el mismo
presupuesto de solver. El gráfico de tiempo permite examinar también la
preparación, que no es gratuita; estos límites no son plazos operacionales
garantizados.

## Cómo incluirlo en el paper

Después del análisis de la Figura 5, añadiría una subsección breve sobre
inicialización heurística. Usaría `warm1` para presentar la calidad disponible
con los dos greedy y las tres variantes de HiGHS. Añadiría una tabla pequeña
con las dos mejoras anteriores y el número de instancias que mejoran. `warm2`
permite comprobar visualmente esa atribución; `warm3` muestra el coste temporal.
No pondría las cuatro vistas del warm start en el cuerpo del paper.

La traza `warm4` es la opción más intuitiva para explicar el mecanismo a los
profesores: muestra dónde parte cada plan y cuándo mejora durante una ejecución.
Es una ilustración de una instancia, no sustituye la comparación de 30 réplicas.
Los puntos de los antiguos CSV provienen de reinicios independientes; no se
deben presentar como una traza de una sola búsqueda.

## Lo que realmente se comprobó

El ejemplo publicado usa 20 tareas y la semilla 1 en A y B, a 1, 5, 15 y
60 segundos de presupuesto. Incluye 24 ejecuciones de solver y cuatro
mediciones greedy. En A, todos los tratamientos alcanzaron el mismo óptimo a
15 y 60 segundos. En B, los warm starts conservaron sus greedy sin mejora y
los cold no devolvieron incumbent. En este piloto, el aporte de inyectar el
greedy frente a conservarlo fuera del solver fue cero en todos los presupuestos.
No es una conclusión sobre las 30 réplicas. Los tiempos reales de B también
superaron el límite nominal de 60 segundos en esta ejecución.

Para resultados publicables, falta ejecutar las cinco cantidades de tareas y
30 semillas con los tres tratamientos, en el mismo entorno. El README incluye
las instrucciones y reanudación. Conviene comenzar con tres semillas, revisar
tiempos y memoria, y ampliar después. La interfaz nativa cambia el orden de
construcción frente al wrapper histórico: hay que repetir el control cold.

El objetivo sigue teniendo un propósito principal por tarea. Las cantidades
auxiliares consumen recursos sin recibir un beneficio independiente. El warm
start no resuelve ni modifica esa elección conceptual; cambiarla exigiría
recalcular las soluciones y las referencias óptimas.

Un incumbent anterior puede reutilizarse en **la misma instancia** mediante
las opciones de replay existentes. Para estudiar una estrategia en dos etapas,
hay que contabilizar el tiempo de la primera y comparar con una búsqueda
continua de igual presupuesto total. Una solución óptima previa sirve para
comprobar la implementación, pero entregarla gratuitamente no representa una
decisión rápida. No se puede trasladar directamente una solución de A a B.

## Ejecutar las opciones

Desde `Resuelto_Paper/TTAP_WarmStart`:

```bash
python visual_options.py --option all --output figures/options
python warm_visual_options.py --warm-csv examples/functional_pilot_n1_20261004/metrics.csv --allow-partial --trace-csv examples/functional_pilot_n1_20261004/trace.csv --trace-scenario A4x4_T20_S1 --trace-budget 60 --output figures/pilot
```

Para una única alternativa, sustituir `all` por `1`, `2`, `3`, `4` o `5`.
La celda de parámetros de `20261004_Figure5_Visual_Options.ipynb` permite elegir
la misma opción. Las corridas nuevas están desactivadas por defecto.
