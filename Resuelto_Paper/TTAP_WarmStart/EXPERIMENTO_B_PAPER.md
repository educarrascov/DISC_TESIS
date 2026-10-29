# Experimento B para la sección 4.2

El perfil `TTAP-B-paper-six-budgets-v1` usa el generador B original: seis
helicópteros heterogéneos, siete áreas y 20/30/40/50/60 tareas, con 30 semillas
por tamaño. El objetivo multidimensional, pesos 2/1.5/1 y referencias 3/3/500
coinciden con el perfil A. Las referencias son escalas de demanda del generador,
no las capacidades máximas de esta flota heterogénea.

## Ejecutar

1. Descargar `20261028_Experimento_B_WarmStart.ipynb` de la revisión publicada.
2. Reiniciar el kernel y ejecutar la carga verificada.
3. Mantener `MODE='pilot'` para comprobar una instancia de 20 tareas.
4. Si el piloto termina y exporta sin errores, cambiar a `MODE='full'`.
5. Conservar la carpeta de resultados y `RESUME=True` para reanudar.

Cold, Benefit start y Density start reciben, cada uno, 1/5/15/60/600/1800 s.
Cada presupuesto inicia una búsqueda nueva. No se suministran óptimos de
corridas largas como starts de corridas cortas. No existe una fase adicional
de referencia: las corridas de 600 y 1800 s ya están incluidas.

| Modo | Instancias | Optimizaciones | Greedy | Suma de presupuestos |
|---|---:|---:|---:|---:|
| pilot | 1 | 18 | 2 | 2.0675 h |
| screen | 9 | 162 | 18 | 18.6075 h |
| full | 150 | 2700 | 300 | 310.125 h |

Estas sumas no predicen la duración real. Una búsqueda puede terminar antes
al certificar; generación, validación, escritura y excesos del límite agregan
tiempo. B puede tardar mucho más que A. Registrar CPU y RAM, mantener el equipo
conectado y evitar suspensión. No ejecutar simultáneamente otro experimento
que compita por CPU o memoria si se van a comparar tiempos.

La salida es `ttap_runs/B_paper_<modo>_<huella>`. Las corridas B anteriores de
cuatro presupuestos y referencias sólo cold quedan separadas. No se anexan
automáticamente al nuevo protocolo. El modo full produce 3000 filas: 2700
optimizaciones y 300 greedy.

## Salidas del paper

`paper` contiene las tablas 3 y 4, las figuras B1/B8, las figuras B3/B9--B13
por presupuesto y `Section_4_2.tex`. Un piloto escribe `paper_preview`, lleva
marca visible y no genera una sección definitiva.

Las tablas principales comparan los cinco métodos a 1800 s sobre las mismas
150 instancias, incluidos incumbents no certificados. No se restringen a los
casos fáciles que lograron certificar. Si cold no retorna un plan, el valor
disponible se puntúa por separado como idle=0; el incumbent sigue ausente.
Las brechas al óptimo usan sólo referencias certificadas del mismo modelo y
objetivo, con su número de observaciones. Un mejor valor conocido sin prueba
de optimalidad no se denomina óptimo. Para comparar A y B bajo igual
presupuesto, usar el punto común de 600 s, además del benchmark B de 1800 s.

Los PDF y SVG son vectoriales. El PNG usa 1200 dpi por defecto. El notebook
muestra SVG. Líneas más finas, marcadores pequeños y bandas tenues mejoran
la lectura sin interpolar ni modificar los datos. Reexportar no ejecuta HiGHS.

Antes de integrar B en el manuscrito, comprobar sus 150 instancias, comparar
los CSV con las tablas y actualizar cualquier frase que aún describa el
objetivo basado sólo en la clase principal. Los resultados del piloto no
permiten anticipar las conclusiones de la corrida completa.


B14 usa las trazas de cada corrida de 1800 s para reconstruir calidad observada
por reloj real. No usa resultados futuros ni une los reinicios. El greedy se
acredita por separado al inicio y la preparación queda fuera del reloj. B3/B9--B13
siguen representando límites solicitados a HiGHS, que pueden excederse.
