# Especificación del TTAP–MDP v1.0

## 1. Proceso de decisión

El controlador es centralizado y observa el sistema en épocas de decisión
determinadas por despachos, término de tareas, recuperación, fin de
indisponibilidad y revelación de demandas:

\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma,T).
\]

El horizonte es finito y se utiliza \(\gamma=1\).

## 2. Estado

El estado contiene:

- tiempo actual y beneficio acumulado;
- por helicóptero: nodo, recursos remanentes, estado operacional, instante de
  disponibilidad, capacidad visual y tarea activa;
- por tarea: estado de ciclo de vida, nodo, recursos requeridos, umbrales
  temporales, prioridad, instante de revelación, helicóptero asignado e
  instante de término.

Los estados de helicóptero son `AVAILABLE`, `BUSY` y `UNAVAILABLE`. Los estados
de tarea son `UNREVEALED`, `PENDING`, `ASSIGNED`, `COMPLETED` y `EXPIRED`.

## 3. Acciones

El catálogo discreto es fijo:

\[
\mathcal{A}=
\{\mathrm{WAIT}\}
\cup\{\mathrm{RETURN}(h):h\in H\}
\cup\{\mathrm{ASSIGN}(h,i):h\in H,i\in I\}.
\]

La máscara permite `ASSIGN(h,i)` solo cuando:

- \(h\) está disponible;
- \(i\) está revelada y pendiente;
- existe compatibilidad visual;
- los recursos remanentes cubren el requerimiento;
- la tarea conserva beneficio positivo;
- nominalmente es posible terminar la tarea y efectuar la recuperación final
  dentro del horizonte.

`RETURN(h)` es factible para un helicóptero disponible fuera de base cuando su
retorno y recuperación caben nominalmente en el horizonte. `WAIT` siempre está
disponible mientras el episodio no haya terminado.

## 4. Transiciones

Un despacho exitoso consume recursos cuando termina la tarea. El helicóptero
queda disponible en el nodo de demanda y puede ejecutar otra tarea. La acción
de retorno incluye vuelo a base y el tiempo de recuperación \(\rho\); después
restaura la capacidad completa.

El tiempo se discretiza en incrementos de \(\Delta t\) redondeando cada duración
hacia arriba, igual que en la instancia determinista.

### Incertidumbre

- El tiempo de viaje usa un multiplicador lognormal de media uno parametrizado
  por su coeficiente de variación.
- Antes de un despacho puede ocurrir una pérdida temporal de disponibilidad.
  La tarea permanece pendiente y el helicóptero vuelve a estar disponible
  después del tiempo de recuperación muestreado.
- Las tareas no iniciales se revelan durante una ventana temporal configurable.

La configuración determinista fija variabilidad y probabilidad de falla en
cero y revela todas las tareas en \(t=0\).

## 5. Recompensa

La recompensa solo se entrega al completar una tarea:

\[
r_k=\frac{w_iS_i(C_i)}{\sum_{j\in I}w_j}.
\]

\(S_i(C_i)\) es la satisfacción temporal anclada en 1, 0.5 y 0 para los
umbrales óptimo, efectivo e inefectivo. No se agregan costos de vuelo ni
penalizaciones a la recompensa científica base. Para DQN, una acción inválida
se comporta como `WAIT` con recompensa cero.

## 6. Terminación

El episodio termina al alcanzar el horizonte o cuando todas las tareas están
completadas/expiradas y todos los helicópteros están disponibles en base con
capacidad completa.

## 7. Comparadores

- `RandomFeasiblePolicy`: muestreo uniforme entre acciones factibles.
- `GreedyOnlinePolicy`: reproduce la regla asíncrona y el orden de desempate del
  primer paper.
- `RollingHorizonPolicy`: emparejamiento exacto de helicópteros y tareas en la
  época actual, resuelto mediante programación dinámica.
- `MaskablePPO`: política RL principal con máscara.
- `DQN`: segunda arquitectura RL, sin máscara nativa.

La v1.0 reproduce el beneficio Greedy 0.593077 del caso Talcahuano antes de
activar incertidumbre.
