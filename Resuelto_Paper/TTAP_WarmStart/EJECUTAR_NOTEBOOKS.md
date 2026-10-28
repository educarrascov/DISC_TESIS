# Ejecutar los notebooks A/B

Los notebooks `20261028_Experimento_A_WarmStart.ipynb` y
`20261028_Experimento_B_WarmStart.ipynb` se encuentran en `Resuelto_Paper`.
Ejecutan nuevas instancias del MILP con tres tratamientos: sin solución
inicial, con Greedy Benefit y con Greedy Density. No cargan resultados
terminados para sustituir la optimización.

Abrir cada notebook en Jupyter o Colab y ejecutar las celdas en orden.
El modo inicial `pilot` usa 20 tareas, semilla 1 y presupuestos 1/5/15/60 s.
El modo `full` usa los cinco tamaños originales y semillas 1 a 30.
Mantener una carpeta persistente y el mismo entorno para reanudar.
La configuración y las fuentes se comprueban antes de reutilizar checkpoints.

`OBJECTIVE_MODE="multidimensional"` interpreta las cantidades como outputs
distintos de servicio médico, transporte no médico y cargo. `primary`
reproduce la definición histórica con un output principal. Los pesos y las
referencias se fijan antes de construir los greedy y el MILP.

El código comprueba el vector inicial completo y lo suministra a HiGHS sin
fijar sus decisiones posteriores. El control cold se vuelve a ejecutar con
la misma interfaz y los mismos parámetros. Los gráficos distinguen calidad,
certificación, mejora sobre el greedy y aporte de suministrarlo al solver.

Las referencias óptimas históricas sólo se leen para el objetivo original
sin cambios. Los archivos públicos se obtienen del commit base indicado y
se verifican mediante su identificador Git. Para un objetivo distinto se
necesitan certificados nuevos; los presupuestos largos se pueden ejecutar
en una fase posterior, con el mismo manifiesto y la misma carpeta.

La suma de los presupuestos cortos del estudio completo es 10.125 horas por
configuración. Las fases largas agregan presupuestos nominales y pueden
ser costosas. El notebook muestra esa carga antes de resolver. Los límites
del solver no son plazos garantizados de respuesta total.

Entorno de referencia, con Python 3.12:

```bash
python -m pip install -r requirements_execution.txt
python -m unittest discover -p 'test_*.py' -v
```

El código publicado no incluye nuevos resultados, conjuntos GPS, manuscritos
ni comentarios de revisión. Los datos originales permanecen en sus rutas
publicadas; cada usuario guarda sus nuevas corridas localmente.
