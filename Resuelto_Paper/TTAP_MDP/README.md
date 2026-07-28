# TTAP–MDP v1.0

Versión ejecutable de la extensión dinámica y estocástica del **Target and
Task Assignment Problem (TTAP)**. El proyecto conserva el beneficio temporal,
las capacidades heterogéneas, el consumo de recursos y la recuperación en base
del modelo determinista del primer paper, y los expone como un MDP centralizado.

## Qué incluye

- escenario pequeño de validación;
- caso `GPS_7areas_6helos_30tasks` de Talcahuano;
- simulador determinista y estocástico dirigido por eventos;
- tiempos de viaje lognormales;
- indisponibilidad operacional y recuperación de helicópteros;
- revelación dinámica de tareas;
- acciones `ASSIGN(h,i)`, `RETURN(h)` y `WAIT`;
- máscara de acciones factibles;
- beneficio temporal normalizado del primer paper;
- políticas Random factible, Greedy online y horizonte móvil;
- adaptador Gymnasium;
- entrenamiento MaskablePPO y DQN;
- notebook principal con resultados;
- pruebas automáticas y resultados CSV reproducibles.

## Inicio rápido en VS Code

Abre **esta carpeta completa** en VS Code. Desde su raíz:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
python -m unittest discover -s tests -v
```

En Windows, activa el entorno con:

```powershell
.venv\Scripts\activate
```

Luego abre:

```text
notebooks/00_TTAP_MDP_v1_results.ipynb
```

Selecciona el kernel `.venv` y ejecuta **Restart Kernel and Run All Cells**.

También puedes obtener resultados inmediatamente desde la terminal, sin
Jupyter ni dependencias externas:

```bash
python -m ttap_mdp.demo --scenario talcahuano --episodes 30
python -m ttap_mdp.demo --scenario talcahuano --episodes 30 --stochastic
```

## Puente determinista validado

Con incertidumbre desactivada, la política Greedy reproduce el caso publicado:

| Métrica | Resultado |
|---|---:|
| Beneficio acumulado | 0.593077 |
| Tareas completadas | 26 / 30 |
| Tareas no completadas | T3, T11, T23, T29 |
| Makespan con retorno final | 117 min |
| Tiempo total de vuelo | 126.644743 min |

Esto comprueba que el nuevo simulador parte de las reglas deterministas
validadas antes de activar llegadas, fallas o tiempos inciertos.

## Entrenamiento de RL

Instala las dependencias completas:

```bash
python -m pip install -e ".[all]"
```

Valida el adaptador Gymnasium:

```bash
python -m ttap_mdp.training.validate_environment
```

Entrena primero en el escenario pequeño:

```bash
python -m ttap_mdp.training.train_ppo \
  --scenario small --stochastic --timesteps 100000

python -m ttap_mdp.training.train_dqn \
  --scenario small --stochastic --timesteps 150000
```

Después se puede cambiar `small` por `talcahuano`. MaskablePPO es el método RL
principal porque utiliza la máscara de factibilidad. DQN se mantiene como
comparador; como DQN estándar no consume máscaras, una acción inválida se
interpreta como `WAIT` y conserva recompensa cero.

Evalúa un modelo guardado sobre 30 semillas:

```bash
python -m ttap_mdp.training.evaluate_rl \
  models/maskable_ppo_small_stochastic.zip \
  --algorithm ppo --scenario small --stochastic --episodes 30
```

## Estructura

```text
TTAP_MDP/
├── notebooks/
│   ├── 00_TTAP_MDP_v1_results.ipynb
│   └── 01_validate_deterministic_model.ipynb
├── results/
│   ├── deterministic_baselines.csv
│   └── stochastic_baselines.csv
├── tests/
├── ttap_mdp/
│   ├── baselines/
│   ├── training/
│   ├── action_mask.py
│   ├── actions.py
│   ├── dynamics.py
│   ├── entities.py
│   ├── environment.py
│   ├── evaluation.py
│   ├── rewards.py
│   └── scenario.py
├── MDP_SPECIFICATION.md
├── pyproject.toml
└── requirements.txt
```

## Alcance científico de esta versión

La v1.0 permite producir resultados deterministas y estocásticos, entrenar PPO
y DQN, y comparar políticas bajo semillas comunes. El horizonte móvil resuelve
exactamente el emparejamiento binario de despacho de cada época; no sustituye
al MILP state–arc completo del primer paper. Las probabilidades de falla y las
distribuciones incluidas son parámetros experimentales iniciales, no
estimaciones empíricas de la Armada ni del evento de 2010.
