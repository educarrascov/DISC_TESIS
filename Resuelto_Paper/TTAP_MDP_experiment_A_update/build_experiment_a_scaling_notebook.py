"""Build the Experiment-A scaling notebook for Greedy online versus PPO."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def _source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    markdown(
        r"""
        # Experimento A dinámico: Greedy online vs. MaskablePPO

        Este notebook amplía la comparación a la familia de escalamiento:

        \[
        4\ \text{helicópteros},\quad 4\ \text{áreas},\quad
        |I|\in\{20,30,40,50,60\}.
        \]

        La geometría, las capacidades, las velocidades, la exigencia visual y el
        generador anidado de tareas son los mismos del Experimento A. Se comparan
        únicamente **Greedy online** y **MaskablePPO** dentro del MDP estocástico.
        El MILP no participa en este experimento.

        La curva de aprendizaje es solo un diagnóstico. La evidencia principal es
        el desempeño de la política congelada sobre 30 instancias que PPO no vio
        durante el entrenamiento.
        """
    ),
    markdown(
        r"""
        ## Qué responde cada bloque

        | Bloque | Pregunta |
        |---|---|
        | Curva de entrenamiento | ¿PPO mejora mientras actualiza sus pesos? |
        | Evaluación fuera de muestra | ¿La política aprendida funciona en instancias nuevas? |
        | Diferencia pareada | ¿Cuánto gana o pierde PPO frente a Greedy en la misma instancia? |
        | Semillas de entrenamiento | ¿El resultado depende de una inicialización afortunada? |
        | Métricas operacionales | ¿La ventaja viene de prioridad, oportunidad, cobertura o vuelo? |
        | Episodios detallados | ¿Qué decisiones concretas hacen divergir a las políticas? |

        Una curva ascendente **no demuestra** que PPO supere a Greedy. Solo la
        evaluación fuera de muestra permite hacer esa comparación.
        """
    ),
    code(
        r"""
        from pathlib import Path
        import platform
        import sys
        import time
        import warnings

        project_root = Path.cwd().resolve()
        if project_root.name == "notebooks":
            project_root = project_root.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        from ttap_mdp import (
            EXPERIMENT_A_GENERATOR_VERSION,
            EXPERIMENT_A_INSTANCE_SEEDS,
            EXPERIMENT_A_TASK_COUNTS,
            ExperimentAFamilyEnv,
            TTAPGymEnv,
            UncertaintyConfig,
            build_experiment_a_scenario,
            run_episode,
        )
        from ttap_mdp.baselines import GreedyOnlinePolicy

        warnings.filterwarnings("ignore", category=FutureWarning)
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.width", 180)

        print("Python:", sys.version.split()[0])
        print("Platform:", platform.platform())
        print("Project root:", project_root)
        print("Generator:", EXPERIMENT_A_GENERATOR_VERSION)
        """
    ),
    markdown(
        r"""
        ## 1. Panel de control

        Ejecuta primero `MODE = "pilot"`. Este modo recorre los cinco tamaños,
        entrena una política por tamaño y evalúa las 30 instancias, pero con menos
        pasos y una sola semilla PPO.

        Cuando el piloto esté revisado, cambia a `MODE = "formal"`:

        - tres semillas independientes de entrenamiento;
        - más pasos por política;
        - cinco realizaciones estocásticas por instancia;
        - 30 instancias de evaluación por tamaño.

        Los modelos y CSV se guardan después de cada etapa, de modo que puedes
        detener y continuar sin perder el progreso.
        """
    ),
    code(
        r"""
        MODE = "pilot"                 # "pilot" o "formal"
        RETRAIN_MISSING_ONLY = True    # reutiliza modelos ya guardados
        RUN_TRAINING = True
        RUN_EVALUATION = True

        TASK_SIZES = list(EXPERIMENT_A_TASK_COUNTS)
        TRAIN_INSTANCE_SEEDS = list(range(1001, 1061))
        EVAL_INSTANCE_SEEDS = list(EXPERIMENT_A_INSTANCE_SEEDS)

        if MODE == "pilot":
            PPO_TRAIN_SEEDS = [218]
            TRAIN_TIMESTEPS = 50_000
            EVAL_REPLICATES = 1
        elif MODE == "formal":
            PPO_TRAIN_SEEDS = [218, 219, 220]
            TRAIN_TIMESTEPS = 300_000
            EVAL_REPLICATES = 5
        else:
            raise ValueError("MODE debe ser 'pilot' o 'formal'.")

        uncertainty = UncertaintyConfig.moderate()
        model_dir = project_root / "models" / "experiment_a_scaling"
        result_dir = project_root / "results" / "experiment_a_scaling"
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        evaluation_path = result_dir / f"evaluation_{MODE}.csv"
        training_index_path = result_dir / f"training_index_{MODE}.csv"

        experiment_config = pd.DataFrame(
            [
                {
                    "Mode": MODE,
                    "Task sizes": str(TASK_SIZES),
                    "Helicopters": 4,
                    "Demand areas": 4,
                    "Training instances": len(TRAIN_INSTANCE_SEEDS),
                    "Evaluation instances": len(EVAL_INSTANCE_SEEDS),
                    "PPO training seeds": str(PPO_TRAIN_SEEDS),
                    "Timesteps per PPO": TRAIN_TIMESTEPS,
                    "Stochastic replicates": EVAL_REPLICATES,
                    "Travel CV": uncertainty.travel_time_cv,
                    "Failure probability": uncertainty.failure_probability,
                    "Arrival window": uncertainty.dynamic_arrival_window,
                    "Initially visible": uncertainty.initial_task_fraction,
                }
            ]
        )
        experiment_config
        """
    ),
    markdown(
        r"""
        ### Separación entrenamiento–evaluación

        - PPO aprende usando instancias `1001–1060`.
        - La comparación final usa exclusivamente las instancias `1–30`.
        - Durante la evaluación los pesos de PPO permanecen congelados.

        Esta separación evita medir memorización. Las cinco cargas de una misma
        semilla son anidadas: la instancia de 20 tareas es el prefijo de la de 60.
        """
    ),
    code(
        r"""
        assert set(TRAIN_INSTANCE_SEEDS).isdisjoint(EVAL_INSTANCE_SEEDS)
        assert TASK_SIZES == [20, 30, 40, 50, 60]
        assert EVAL_INSTANCE_SEEDS == list(range(1, 31))

        example = build_experiment_a_scenario(20, 1)
        nodes_table = pd.DataFrame(
            [
                {
                    "Node": node.node_id,
                    "x (km)": node.first_coordinate,
                    "y (km)": node.second_coordinate,
                    "Role": "Base" if node.is_base else "Demand area",
                }
                for node in example.nodes
            ]
        )
        fleet_table = pd.DataFrame(
            [
                {
                    "Helicopter": helicopter.helicopter_id,
                    "Visual": helicopter.visual_capable,
                    "Cargo": helicopter.capacity.cargo,
                    "Medical": helicopter.capacity.medical,
                    "Personnel": helicopter.capacity.personnel,
                    "Speed (km/h)": helicopter.speed_kmh,
                }
                for helicopter in example.helicopters
            ]
        )
        display(nodes_table)
        display(fleet_table)
        """
    ),
    code(
        r"""
        validation_rows = []
        for instance_seed in EVAL_INSTANCE_SEEDS:
            scenario_60 = build_experiment_a_scenario(60, instance_seed)
            for n_tasks in TASK_SIZES:
                scenario_n = build_experiment_a_scenario(n_tasks, instance_seed)
                assert scenario_n.tasks == scenario_60.tasks[:n_tasks]
                counts = pd.Series(
                    [task.node_id for task in scenario_n.tasks]
                ).value_counts()
                validation_rows.append(
                    {
                        "Tasks": n_tasks,
                        "Instance seed": instance_seed,
                        "All areas represented": set(counts.index)
                        == {"A", "B", "C", "D"},
                        "Largest-smallest area load": int(
                            counts.max() - counts.min()
                        ),
                    }
                )

        validation = pd.DataFrame(validation_rows)
        assert validation["All areas represented"].all()
        print("Validated nested scenarios:", len(validation))
        validation.groupby("Tasks").agg(
            instances=("Instance seed", "nunique"),
            minimum_imbalance=("Largest-smallest area load", "min"),
            maximum_imbalance=("Largest-smallest area load", "max"),
        )
        """
    ),
    markdown(
        r"""
        ## 2. Por qué se entrena un PPO por tamaño

        En esta implementación el vector de estado contiene nueve valores por
        tarea y el catálogo contiene una asignación por par
        helicóptero–tarea:

        \[
        |\text{obs}|=2+7|H|+9|I|,\qquad
        |\mathcal A|=1+|H|+|H||I|.
        \]

        Por ello las redes de 20 y 60 tareas tienen entradas y salidas de tamaños
        distintos. Entrenar cinco PPO independientes es la opción más transparente
        para este primer experimento de escalamiento; no se ocultan tareas mediante
        relleno y cada PPO se compara con Greedy en su mismo problema.
        """
    ),
    code(
        r"""
        dimension_rows = []
        for n_tasks in TASK_SIZES:
            scenario = build_experiment_a_scenario(n_tasks, 1)
            environment = TTAPGymEnv(scenario, uncertainty)
            dimension_rows.append(
                {
                    "Tasks": n_tasks,
                    "Observation values": environment.observation_space.shape[0],
                    "Discrete actions": environment.action_space.n,
                }
            )
            environment.close()
        pd.DataFrame(dimension_rows)
        """
    ),
    markdown(
        r"""
        ## 3. Entrenamiento PPO sobre una familia de instancias

        En cada episodio de entrenamiento se selecciona una instancia diferente
        del conjunto `1001–1060`. La política no se entrena contra una sola tabla
        fija de tareas.

        Se registra la curva dentro del notebook y se guarda un CSV por modelo.
        TensorBoard no es necesario.
        """
    ),
    code(
        r"""
        try:
            from sb3_contrib import MaskablePPO
            from sb3_contrib.common.wrappers import ActionMasker
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor
        except ImportError as exc:
            raise ImportError(
                'Faltan dependencias RL. Ejecuta: python -m pip install -e ".[all]"'
            ) from exc


        def mask_fn(environment):
            return environment.action_masks()


        class TrainingHistoryCallback(BaseCallback):
            def __init__(self, n_tasks, ppo_seed):
                super().__init__(verbose=0)
                self.n_tasks = int(n_tasks)
                self.ppo_seed = int(ppo_seed)
                self.records = []

            def _on_step(self):
                for info, done in zip(
                    self.locals.get("infos", []),
                    self.locals.get("dones", []),
                ):
                    episode = info.get("episode")
                    if done and episode is not None:
                        self.records.append(
                            {
                                "tasks": self.n_tasks,
                                "ppo_seed": self.ppo_seed,
                                "timestep": int(self.num_timesteps),
                                "episode_return": float(episode["r"]),
                                "episode_length": int(episode["l"]),
                                "instance_seed": int(info["instance_seed"]),
                            }
                        )
                return True


        def model_stem(n_tasks, ppo_seed):
            return model_dir / f"ppo_T{n_tasks}_seed{ppo_seed}"


        def history_path(n_tasks, ppo_seed):
            return result_dir / f"training_T{n_tasks}_seed{ppo_seed}.csv"
        """
    ),
    code(
        r"""
        models = {}
        training_frames = []
        training_index = []

        if RUN_TRAINING:
            total_models = len(TASK_SIZES) * len(PPO_TRAIN_SEEDS)
            completed_models = 0
            for n_tasks in TASK_SIZES:
                for ppo_seed in PPO_TRAIN_SEEDS:
                    destination = Path(f"{model_stem(n_tasks, ppo_seed)}.zip")
                    csv_path = history_path(n_tasks, ppo_seed)
                    start = time.perf_counter()

                    if RETRAIN_MISSING_ONLY and destination.exists():
                        model = MaskablePPO.load(destination, device="cpu")
                        history = (
                            pd.read_csv(csv_path)
                            if csv_path.exists()
                            else pd.DataFrame()
                        )
                        status = "loaded"
                    else:
                        family_environment = ExperimentAFamilyEnv(
                            n_tasks,
                            TRAIN_INSTANCE_SEEDS,
                            uncertainty,
                            sampler_seed=ppo_seed,
                        )
                        training_environment = Monitor(
                            ActionMasker(family_environment, mask_fn)
                        )
                        callback = TrainingHistoryCallback(n_tasks, ppo_seed)
                        model = MaskablePPO(
                            "MlpPolicy",
                            training_environment,
                            seed=ppo_seed,
                            learning_rate=3e-4,
                            n_steps=2048,
                            batch_size=64,
                            gamma=0.99,
                            policy_kwargs={"net_arch": [128, 128]},
                            verbose=0,
                            device="cpu",
                        )
                        model.learn(
                            total_timesteps=TRAIN_TIMESTEPS,
                            callback=callback,
                            progress_bar=True,
                        )
                        model.save(model_stem(n_tasks, ppo_seed))
                        training_environment.close()
                        history = pd.DataFrame(callback.records)
                        history.to_csv(csv_path, index=False)
                        status = "trained"

                    models[(n_tasks, ppo_seed)] = model
                    if not history.empty:
                        training_frames.append(history)
                    completed_models += 1
                    elapsed = time.perf_counter() - start
                    training_index.append(
                        {
                            "tasks": n_tasks,
                            "ppo_seed": ppo_seed,
                            "status": status,
                            "timesteps": TRAIN_TIMESTEPS,
                            "elapsed_seconds": elapsed,
                            "model": str(destination),
                            "history": str(csv_path),
                        }
                    )
                    print(
                        f"[{completed_models}/{total_models}] "
                        f"T={n_tasks}, seed={ppo_seed}: {status} "
                        f"({elapsed:.1f} s)"
                    )

            pd.DataFrame(training_index).to_csv(training_index_path, index=False)
        else:
            for n_tasks in TASK_SIZES:
                for ppo_seed in PPO_TRAIN_SEEDS:
                    destination = Path(f"{model_stem(n_tasks, ppo_seed)}.zip")
                    if not destination.exists():
                        raise FileNotFoundError(destination)
                    models[(n_tasks, ppo_seed)] = MaskablePPO.load(
                        destination, device="cpu"
                    )
                    csv_path = history_path(n_tasks, ppo_seed)
                    if csv_path.exists():
                        training_frames.append(pd.read_csv(csv_path))

        training_history = (
            pd.concat(training_frames, ignore_index=True)
            if training_frames
            else pd.DataFrame()
        )
        print("Models available:", len(models))
        print("Training episodes recorded:", len(training_history))
        """
    ),
    code(
        r"""
        if training_history.empty:
            print("No training history was loaded.")
        else:
            fig, axes = plt.subplots(
                len(TASK_SIZES),
                1,
                figsize=(11, 2.7 * len(TASK_SIZES)),
                sharex=True,
            )
            for ax, n_tasks in zip(axes, TASK_SIZES):
                subset = training_history.loc[
                    training_history["tasks"] == n_tasks
                ]
                for ppo_seed, seed_data in subset.groupby("ppo_seed"):
                    seed_data = seed_data.sort_values("timestep").copy()
                    window = max(10, min(50, len(seed_data) // 10))
                    seed_data["smoothed"] = (
                        seed_data["episode_return"]
                        .rolling(window, min_periods=1)
                        .mean()
                    )
                    ax.plot(
                        seed_data["timestep"],
                        seed_data["smoothed"],
                        linewidth=1.8,
                        label=f"seed {ppo_seed}",
                    )
                ax.set_ylabel(f"T={n_tasks}")
                ax.grid(alpha=0.25)
                ax.legend(loc="lower right")
            axes[0].set_title(
                "Curvas de entrenamiento PPO — diagnóstico, no resultado final"
            )
            axes[-1].set_xlabel("Pasos de entrenamiento")
            fig.text(
                0.01,
                0.5,
                "Retorno medio móvil",
                rotation=90,
                va="center",
            )
            plt.tight_layout(rect=(0.03, 0, 1, 1))
            plt.show()
        """
    ),
    markdown(
        r"""
        ## 4. Evaluación congelada sobre las 30 instancias

        Para cada tamaño:

        1. se construye una de las instancias `1–30`;
        2. Greedy y PPO reciben el mismo escenario;
        3. se usa la misma semilla de episodio;
        4. PPO actúa de forma determinista y no actualiza sus pesos.

        Los flujos aleatorios son contrabalanceados por claves de evento. La
        revelación de una tarea y una misma asignación helicóptero–tarea conservan
        su realización aunque otra política haya ejecutado antes una acción distinta.
        Acciones diferentes, naturalmente, generan eventos diferentes.
        """
    ),
    code(
        r"""
        PRIORITIES = ("medical", "personnel", "cargo")


        def policy_row(
            simulator,
            policy,
            n_tasks,
            instance_seed,
            episode_replicate,
            episode_seed,
            ppo_seed,
            steps,
        ):
            summary = simulator.summary()
            rewards = {
                record.task_id: record.reward
                for record in simulator.log
                if record.event == "task_completed"
            }
            row = {
                "tasks": int(n_tasks),
                "instance_seed": int(instance_seed),
                "replicate": int(episode_replicate),
                "episode_seed": int(episode_seed),
                "policy": policy,
                "ppo_seed": int(ppo_seed),
                "benefit": float(summary["benefit"]),
                "completed_tasks": int(summary["completed_tasks"]),
                "completion_rate": float(summary["completion_rate"]),
                "relevant_completion_rate": float(
                    summary["relevant_completion_rate"]
                ),
                "average_response_time": float(
                    summary["average_response_time"]
                ),
                "makespan": float(summary["makespan"]),
                "flight_time": float(summary["flight_time"]),
                "failures": int(summary["failures"]),
                "invalid_actions": int(summary["invalid_actions"]),
                "steps": int(steps),
            }
            for priority in PRIORITIES:
                priority_tasks = [
                    task
                    for task in simulator.scenario.tasks
                    if task.priority_class.value == priority
                ]
                completed = [
                    task
                    for task in priority_tasks
                    if simulator.task_states[task.task_id].status.value
                    == "completed"
                ]
                row[f"{priority}_completion_rate"] = (
                    len(completed) / len(priority_tasks)
                    if priority_tasks
                    else np.nan
                )
                row[f"{priority}_benefit"] = sum(
                    rewards.get(task.task_id, 0.0)
                    for task in priority_tasks
                )
            return row


        def run_greedy(scenario, episode_seed, **metadata):
            result, simulator = run_episode(
                scenario,
                GreedyOnlinePolicy(),
                uncertainty=uncertainty,
                seed=episode_seed,
            )
            return (
                policy_row(
                    simulator,
                    policy="Greedy online",
                    ppo_seed=-1,
                    steps=result.steps,
                    episode_seed=episode_seed,
                    **metadata,
                ),
                simulator,
            )


        def run_ppo(scenario, model, ppo_seed, episode_seed, **metadata):
            environment = TTAPGymEnv(scenario, uncertainty)
            observation, _ = environment.reset(seed=episode_seed)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                    action_masks=environment.action_masks(),
                )
                observation, _, terminated, truncated, _ = environment.step(
                    int(action)
                )
                steps += 1
            simulator = environment.simulator
            row = policy_row(
                simulator,
                policy="PPO",
                ppo_seed=ppo_seed,
                steps=steps,
                episode_seed=episode_seed,
                **metadata,
            )
            environment.close()
            return row, simulator
        """
    ),
    code(
        r"""
        evaluation_records = []

        if RUN_EVALUATION:
            total_cases = (
                len(TASK_SIZES)
                * len(EVAL_INSTANCE_SEEDS)
                * EVAL_REPLICATES
            )
            completed_cases = 0
            for n_tasks in TASK_SIZES:
                task_start = time.perf_counter()
                for instance_seed in EVAL_INSTANCE_SEEDS:
                    scenario = build_experiment_a_scenario(
                        n_tasks, instance_seed
                    )
                    for replicate in range(EVAL_REPLICATES):
                        episode_seed = (
                            10_000_000
                            + 100_000 * n_tasks
                            + 100 * instance_seed
                            + replicate
                        )
                        metadata = {
                            "n_tasks": n_tasks,
                            "instance_seed": instance_seed,
                            "episode_replicate": replicate,
                        }
                        greedy_row, _ = run_greedy(
                            scenario, episode_seed, **metadata
                        )
                        evaluation_records.append(greedy_row)

                        for ppo_seed in PPO_TRAIN_SEEDS:
                            ppo_row, _ = run_ppo(
                                scenario,
                                models[(n_tasks, ppo_seed)],
                                ppo_seed,
                                episode_seed,
                                **metadata,
                            )
                            evaluation_records.append(ppo_row)

                        completed_cases += 1

                pd.DataFrame(evaluation_records).to_csv(
                    evaluation_path, index=False
                )
                elapsed = time.perf_counter() - task_start
                print(
                    f"T={n_tasks} completed; "
                    f"{completed_cases}/{total_cases} cases; "
                    f"{elapsed:.1f} s"
                )
        else:
            if not evaluation_path.exists():
                raise FileNotFoundError(evaluation_path)
            evaluation_records = pd.read_csv(evaluation_path).to_dict("records")

        evaluation = pd.DataFrame(evaluation_records)
        assert evaluation["invalid_actions"].sum() == 0
        print("Evaluation rows:", len(evaluation))
        print("Saved:", evaluation_path)
        evaluation.head()
        """
    ),
    markdown(
        r"""
        ## 5. Resultado principal: generalización fuera de muestra

        Primero se promedian las realizaciones estocásticas dentro de cada
        instancia. Para PPO también se promedian las semillas de entrenamiento.
        Así, las 30 instancias —no cientos de episodios artificialmente tratados
        como independientes— son la unidad principal de comparación.
        """
    ),
    code(
        r"""
        METRICS = [
            "benefit",
            "completed_tasks",
            "completion_rate",
            "relevant_completion_rate",
            "average_response_time",
            "makespan",
            "flight_time",
            "medical_completion_rate",
            "personnel_completion_rate",
            "cargo_completion_rate",
        ]

        greedy_instance = (
            evaluation.loc[evaluation["policy"] == "Greedy online"]
            .groupby(["tasks", "instance_seed"], as_index=False)[METRICS]
            .mean()
        )
        ppo_model_instance = (
            evaluation.loc[evaluation["policy"] == "PPO"]
            .groupby(
                ["tasks", "instance_seed", "ppo_seed"],
                as_index=False,
            )[METRICS]
            .mean()
        )
        ppo_instance = (
            ppo_model_instance
            .groupby(["tasks", "instance_seed"], as_index=False)[METRICS]
            .mean()
        )

        greedy_instance["policy"] = "Greedy online"
        ppo_instance["policy"] = "PPO"
        policy_instance = pd.concat(
            [greedy_instance, ppo_instance],
            ignore_index=True,
        )

        summary_rows = []
        for (n_tasks, policy), group in policy_instance.groupby(
            ["tasks", "policy"]
        ):
            row = {
                "Tasks": n_tasks,
                "Policy": policy,
                "Instances": group["instance_seed"].nunique(),
            }
            for metric in (
                "benefit",
                "completed_tasks",
                "relevant_completion_rate",
                "average_response_time",
                "flight_time",
            ):
                values = group[metric].astype(float)
                half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_ci95"] = half
            summary_rows.append(row)

        scaling_summary = pd.DataFrame(summary_rows).sort_values(
            ["Tasks", "Policy"]
        )
        scaling_summary
        """
    ),
    code(
        r"""
        policy_order = ["Greedy online", "PPO"]
        colors = {"Greedy online": "#F28E2B", "PPO": "#4E79A7"}

        fig, ax = plt.subplots(figsize=(10, 5.2))
        for policy in policy_order:
            data = scaling_summary.loc[
                scaling_summary["Policy"] == policy
            ].sort_values("Tasks")
            ax.errorbar(
                data["Tasks"],
                data["benefit_mean"],
                yerr=data["benefit_ci95"],
                marker="o",
                markersize=6,
                linewidth=2.2,
                capsize=4,
                color=colors[policy],
                label=policy,
            )
        ax.set(
            title="Resultado principal fuera de muestra",
            xlabel="Número de tareas",
            ylabel="Beneficio normalizado medio (IC 95%)",
            xticks=TASK_SIZES,
        )
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 6. Diferencia pareada PPO − Greedy

        Para cada una de las 30 instancias se calcula:

        \[
        \Delta_s=B_s^{PPO}-B_s^{Greedy}.
        \]

        - \(\Delta_s>0\): PPO obtiene más beneficio.
        - \(\Delta_s=0\): empate.
        - \(\Delta_s<0\): Greedy obtiene más beneficio.

        El intervalo se calcula con *bootstrap* sobre las 30 diferencias de
        instancia. Esto responde directamente si la ventaja se mantiene al cambiar
        la carga de tareas.
        """
    ),
    code(
        r"""
        paired = ppo_instance.merge(
            greedy_instance,
            on=["tasks", "instance_seed"],
            suffixes=("_ppo", "_greedy"),
        )
        paired["delta_benefit"] = (
            paired["benefit_ppo"] - paired["benefit_greedy"]
        )


        def bootstrap_mean_ci(values, seed, n_boot=5000):
            values = np.asarray(values, dtype=float)
            rng = np.random.default_rng(seed)
            samples = rng.choice(
                values,
                size=(n_boot, len(values)),
                replace=True,
            ).mean(axis=1)
            return np.quantile(samples, [0.025, 0.975])


        paired_rows = []
        for n_tasks, group in paired.groupby("tasks"):
            delta = group["delta_benefit"].to_numpy()
            low, high = bootstrap_mean_ci(delta, 218 + int(n_tasks))
            paired_rows.append(
                {
                    "Tasks": int(n_tasks),
                    "Mean PPO-Greedy": delta.mean(),
                    "CI95 low": low,
                    "CI95 high": high,
                    "PPO wins": int((delta > 1e-12).sum()),
                    "Ties": int(np.isclose(delta, 0.0, atol=1e-12).sum()),
                    "Greedy wins": int((delta < -1e-12).sum()),
                    "PPO win rate": float((delta > 1e-12).mean()),
                }
            )
        paired_summary = pd.DataFrame(paired_rows)
        paired_summary
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

        center = paired_summary["Mean PPO-Greedy"]
        lower_error = center - paired_summary["CI95 low"]
        upper_error = paired_summary["CI95 high"] - center
        axes[0].errorbar(
            paired_summary["Tasks"],
            center,
            yerr=np.vstack([lower_error, upper_error]),
            marker="o",
            linewidth=2.2,
            capsize=4,
            color="#4E79A7",
        )
        axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[0].set(
            title="Ventaja media pareada",
            xlabel="Número de tareas",
            ylabel=r"$\Delta$ beneficio: PPO − Greedy (IC 95%)",
            xticks=TASK_SIZES,
        )
        axes[0].grid(alpha=0.25)

        axes[1].bar(
            paired_summary["Tasks"],
            100 * paired_summary["PPO win rate"],
            width=6,
            color="#4E79A7",
            alpha=0.75,
        )
        axes[1].axhline(50, color="black", linestyle="--", linewidth=1)
        axes[1].set(
            title="Frecuencia de victoria de PPO",
            xlabel="Número de tareas",
            ylabel="Instancias con PPO > Greedy (%)",
            xticks=TASK_SIZES,
            ylim=(0, 100),
        )
        axes[1].grid(axis="y", alpha=0.25)

        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 7. ¿De dónde viene la diferencia?

        El beneficio es la métrica principal, pero no basta para explicar la
        conducta. Los paneles siguientes separan:

        - tareas completadas;
        - tareas relevantes completadas;
        - tiempo medio de respuesta;
        - tiempo total de vuelo.

        Esto permite distinguir una mejora humanitaria de un simple aumento de
        actividad operacional.
        """
    ),
    code(
        r"""
        dashboard = [
            ("completed_tasks", "Tareas completadas", True),
            (
                "relevant_completion_rate",
                "Relevant completion rate",
                True,
            ),
            (
                "average_response_time",
                "Tiempo medio de respuesta",
                False,
            ),
            ("flight_time", "Tiempo total de vuelo", False),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (metric, title, higher_is_better) in zip(
            axes.flat, dashboard
        ):
            for policy in policy_order:
                group = policy_instance.loc[
                    policy_instance["policy"] == policy
                ]
                summary = (
                    group.groupby("tasks")[metric]
                    .agg(["mean", "std", "count"])
                    .reindex(TASK_SIZES)
                )
                ci = 1.96 * summary["std"] / np.sqrt(summary["count"])
                ax.errorbar(
                    TASK_SIZES,
                    summary["mean"],
                    yerr=ci,
                    marker="o",
                    capsize=3,
                    linewidth=1.9,
                    color=colors[policy],
                    label=policy,
                )
            ax.set_title(title)
            ax.set_xticks(TASK_SIZES)
            ax.grid(alpha=0.25)
            direction = "↑ mejor" if higher_is_better else "↓ mejor"
            ax.set_xlabel(direction)
        axes[0, 0].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        priority_metrics = [
            ("medical_completion_rate", "Medical"),
            ("personnel_completion_rate", "Personnel"),
            ("cargo_completion_rate", "Cargo"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ax, (metric, label) in zip(axes, priority_metrics):
            for policy in policy_order:
                group = policy_instance.loc[
                    policy_instance["policy"] == policy
                ]
                summary = (
                    group.groupby("tasks")[metric]
                    .agg(["mean", "std", "count"])
                    .reindex(TASK_SIZES)
                )
                ci = 1.96 * summary["std"] / np.sqrt(summary["count"])
                ax.errorbar(
                    TASK_SIZES,
                    summary["mean"],
                    yerr=ci,
                    marker="o",
                    capsize=3,
                    linewidth=1.8,
                    color=colors[policy],
                    label=policy,
                )
            ax.set_title(label)
            ax.set_xlabel("Número de tareas")
            ax.set_xticks(TASK_SIZES)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Completion rate por prioridad")
        axes[0].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 8. Estabilidad entre semillas PPO

        Un solo entrenamiento puede ser afortunado o desfavorable. En modo formal,
        la tabla muestra el beneficio fuera de muestra de cada semilla PPO.
        Resultados similares entre columnas indican estabilidad. Diferencias grandes
        obligan a aumentar semillas o revisar hiperparámetros.
        """
    ),
    code(
        r"""
        ppo_seed_summary = (
            ppo_model_instance
            .groupby(["tasks", "ppo_seed"], as_index=False)
            .agg(
                benefit_mean=("benefit", "mean"),
                benefit_sd=("benefit", "std"),
                completed_mean=("completed_tasks", "mean"),
                response_mean=("average_response_time", "mean"),
            )
        )
        stability_table = ppo_seed_summary.pivot(
            index="tasks",
            columns="ppo_seed",
            values="benefit_mean",
        )
        stability_table
        """
    ),
    code(
        r"""
        if training_history.empty:
            print("Training history unavailable.")
        else:
            training_tail = (
                training_history.sort_values("timestep")
                .groupby(["tasks", "ppo_seed"], as_index=False)
                .tail(100)
                .groupby(["tasks", "ppo_seed"], as_index=False)
                .agg(training_last100=("episode_return", "mean"))
            )
            generalization = ppo_seed_summary.merge(
                training_tail,
                on=["tasks", "ppo_seed"],
                how="left",
            )
            generalization["evaluation_minus_training"] = (
                generalization["benefit_mean"]
                - generalization["training_last100"]
            )
            display(generalization)
        """
    ),
    markdown(
        r"""
        ## 9. Tres casos operacionales

        Para el tamaño seleccionado se inspeccionan:

        - la instancia donde PPO gana más;
        - una instancia cercana a la diferencia mediana;
        - la instancia donde Greedy gana más.

        Las curvas muestran **cuándo** se captura el beneficio. Luego se listan las
        primeras operaciones para identificar dónde divergen las decisiones.
        """
    ),
    code(
        r"""
        DETAIL_TASKS = 60
        DETAIL_PPO_SEED = PPO_TRAIN_SEEDS[0]
        DETAIL_REPLICATE = 0

        detail_model_instances = ppo_model_instance.loc[
            (ppo_model_instance["tasks"] == DETAIL_TASKS)
            & (ppo_model_instance["ppo_seed"] == DETAIL_PPO_SEED)
        ].merge(
            greedy_instance.loc[greedy_instance["tasks"] == DETAIL_TASKS],
            on=["tasks", "instance_seed"],
            suffixes=("_ppo", "_greedy"),
        )
        detail_model_instances["delta"] = (
            detail_model_instances["benefit_ppo"]
            - detail_model_instances["benefit_greedy"]
        )
        median_delta = detail_model_instances["delta"].median()
        selected_instances = {
            "Mayor ventaja PPO": int(
                detail_model_instances.loc[
                    detail_model_instances["delta"].idxmax(),
                    "instance_seed",
                ]
            ),
            "Caso mediano": int(
                detail_model_instances.loc[
                    (
                        detail_model_instances["delta"] - median_delta
                    ).abs().idxmin(),
                    "instance_seed",
                ]
            ),
            "Mayor ventaja Greedy": int(
                detail_model_instances.loc[
                    detail_model_instances["delta"].idxmin(),
                    "instance_seed",
                ]
            ),
        }
        selected_instances
        """
    ),
    code(
        r"""
        detailed_runs = {}
        detailed_rows = []
        for case_name, instance_seed in selected_instances.items():
            scenario = build_experiment_a_scenario(
                DETAIL_TASKS, instance_seed
            )
            episode_seed = (
                10_000_000
                + 100_000 * DETAIL_TASKS
                + 100 * instance_seed
                + DETAIL_REPLICATE
            )
            metadata = {
                "n_tasks": DETAIL_TASKS,
                "instance_seed": instance_seed,
                "episode_replicate": DETAIL_REPLICATE,
            }
            greedy_row, greedy_simulator = run_greedy(
                scenario, episode_seed, **metadata
            )
            ppo_row, ppo_simulator = run_ppo(
                scenario,
                models[(DETAIL_TASKS, DETAIL_PPO_SEED)],
                DETAIL_PPO_SEED,
                episode_seed,
                **metadata,
            )
            detailed_runs[case_name] = {
                "Greedy online": greedy_simulator,
                "PPO": ppo_simulator,
            }
            detailed_rows.extend(
                [
                    {"case": case_name, **greedy_row},
                    {"case": case_name, **ppo_row},
                ]
            )

        pd.DataFrame(detailed_rows)[
            [
                "case",
                "instance_seed",
                "policy",
                "benefit",
                "completed_tasks",
                "relevant_completion_rate",
                "average_response_time",
                "flight_time",
            ]
        ]
        """
    ),
    code(
        r"""
        def cumulative_trace(simulator):
            times = [0.0]
            values = [0.0]
            cumulative = 0.0
            for record in sorted(simulator.log, key=lambda item: item.time):
                if record.event == "task_completed":
                    cumulative += record.reward
                    times.append(record.time)
                    values.append(cumulative)
            return np.asarray(times), np.asarray(values)


        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
        for ax, (case_name, simulations) in zip(
            axes, detailed_runs.items()
        ):
            for policy in policy_order:
                times, values = cumulative_trace(simulations[policy])
                ax.step(
                    times,
                    values,
                    where="post",
                    linewidth=2,
                    color=colors[policy],
                    label=policy,
                )
            ax.set_title(case_name)
            ax.set_xlabel("Tiempo de misión")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Beneficio acumulado")
        axes[0].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        def operation_sequence(simulator, policy, case_name):
            rows = []
            for record in sorted(
                simulator.log,
                key=lambda item: (
                    item.time,
                    item.helicopter_id or "",
                    item.task_id or "",
                ),
            ):
                if record.event not in (
                    "task_dispatched",
                    "return_dispatched",
                    "dispatch_failure",
                ):
                    continue
                rows.append(
                    {
                        "case": case_name,
                        "policy": policy,
                        "time": record.time,
                        "event": record.event,
                        "helicopter": record.helicopter_id,
                        "task": record.task_id,
                        "node": record.node_id,
                    }
                )
            return pd.DataFrame(rows)


        operation_tables = []
        for case_name, simulations in detailed_runs.items():
            for policy in policy_order:
                table = operation_sequence(
                    simulations[policy], policy, case_name
                )
                table["decision_number"] = np.arange(1, len(table) + 1)
                operation_tables.append(table)

        operations = pd.concat(operation_tables, ignore_index=True)
        print("Primeras 15 operaciones de cada política y caso")
        operations.groupby(["case", "policy"], sort=False).head(15)
        """
    ),
    markdown(
        r"""
        ## 10. Criterio de lectura final

        PPO aporta evidencia útil solo si se cumplen simultáneamente estas condiciones:

        1. la curva de entrenamiento no colapsa;
        2. la política congelada mejora fuera de muestra;
        3. el IC 95% de \(\Delta\) no depende de una única instancia;
        4. el resultado es razonablemente estable entre semillas PPO;
        5. la ventaja puede explicarse mediante prioridad, oportunidad o cobertura;
        6. el costo operacional adicional, si existe, queda cuantificado.

        Si PPO solo muestra una curva ascendente, todavía no existe una conclusión
        comparativa. Si gana fuera de muestra pero solo con una semilla, la evidencia
        sigue siendo exploratoria. El modo formal está diseñado para resolver ambos
        problemas.
        """
    ),
    code(
        r"""
        print("Final validation")
        print("----------------")
        print("Task sizes:", sorted(evaluation["tasks"].unique()))
        print(
            "Evaluation instances per size:",
            evaluation.groupby("tasks")["instance_seed"].nunique().to_dict(),
        )
        print("Policies:", sorted(evaluation["policy"].unique()))
        print("PPO seeds:", sorted(
            evaluation.loc[evaluation["policy"] == "PPO", "ppo_seed"]
            .unique()
            .tolist()
        ))
        print("Invalid actions:", int(evaluation["invalid_actions"].sum()))
        print("Evaluation CSV:", evaluation_path)
        print("Model directory:", model_dir)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (Python 3)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = (
    Path(__file__).resolve().parent
    / "notebooks"
    / "02_experiment_a_scaling_greedy_vs_ppo.ipynb"
)
destination.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print(destination)

