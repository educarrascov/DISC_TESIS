"""Build the self-contained Greedy online versus PPO notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def _source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


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
        # TTAP–MDP: Greedy online vs. MaskablePPO

        Este notebook deja una sola comparación:

        - **Greedy online:** decide con una regla fija de beneficio inmediato.
        - **MaskablePPO:** aprende una política mediante episodios simulados.

        Ambos actúan sobre el **mismo MDP**, usan el mismo catálogo de acciones
        (`ASSIGN`, `RETURN`, `WAIT`) y respetan la misma máscara de factibilidad.
        El MILP del primer paper no participa en este experimento.

        El flujo completo queda visible aquí:

        1. definir escenario e incertidumbre;
        2. entrenar o cargar PPO;
        3. ver la curva de aprendizaje sin TensorBoard;
        4. evaluar Greedy y PPO con las mismas semillas;
        5. comparar distribuciones y métricas;
        6. inspeccionar un episodio tarea por tarea.
        """
    ),
    markdown(
        r"""
        ## 1. ¿Qué diferencia a las dos políticas?

        | Elemento | Greedy online | MaskablePPO |
        |---|---|---|
        | Regla de decisión | Mayor beneficio inmediato factible | Red neuronal entrenada |
        | Entrenamiento | No requiere | Sí, mediante simulación |
        | Información utilizada | Estado disponible ahora | Estado disponible ahora |
        | Consideración del futuro | No explícita | Puede aprender efectos futuros |
        | Acción imposible | Se descarta | Se descarta mediante máscara |

        La comparación responde: **¿el aprendizaje permite tomar mejores decisiones
        secuenciales que la regla Greedy bajo incertidumbre?**
        """
    ),
    code(
        r"""
        from pathlib import Path
        import platform
        import sys
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
            UncertaintyConfig,
            build_small_scenario,
            build_talcahuano_scenario,
            run_episode,
        )
        from ttap_mdp.baselines import GreedyOnlinePolicy
        from ttap_mdp.training.common import make_environment

        warnings.filterwarnings("ignore", category=FutureWarning)
        pd.set_option("display.max_columns", 30)
        pd.set_option("display.width", 140)

        print("Python:", sys.version.split()[0])
        print("Platform:", platform.platform())
        print("Project root:", project_root)
        """
    ),
    markdown(
        r"""
        ## 2. Panel de control

        Esta es la única celda que necesitas modificar normalmente.

        - Comienza con `small` y `20_000` pasos para comprender el flujo.
        - Usa `RETRAIN_PPO = False` después de guardar un modelo que quieras reutilizar.
        - Cambia a `talcahuano` solamente cuando el experimento pequeño esté claro;
          PPO tendrá que entrenarse nuevamente porque cambia el tamaño del estado y de
          las acciones.
        """
    ),
    code(
        r"""
        SCENARIO_NAME = "small"       # "small" o "talcahuano"
        STOCHASTIC = True             # comparación dinámica/estocástica
        TRAIN_TIMESTEPS = 20_000      # luego: 100_000 o más
        TRAIN_SEED = 218
        EVAL_EPISODES = 100
        RETRAIN_PPO = True            # False = cargar el modelo guardado
        SAVE_CSV = True
        SMOOTHING_WINDOW = 25

        uncertainty = (
            UncertaintyConfig.moderate()
            if STOCHASTIC
            else UncertaintyConfig.deterministic()
        )
        scenario = (
            build_talcahuano_scenario()
            if SCENARIO_NAME == "talcahuano"
            else build_small_scenario()
        )
        suffix = "stochastic" if STOCHASTIC else "deterministic"
        model_dir = project_root / "models" / "notebook_ppo"
        model_stem = model_dir / (
            f"maskable_ppo_{SCENARIO_NAME}_{suffix}_seed{TRAIN_SEED}"
        )
        model_path = Path(f"{model_stem}.zip")
        results_path = (
            project_root
            / "results"
            / f"greedy_vs_ppo_{SCENARIO_NAME}_{suffix}.csv"
        )

        config = pd.DataFrame(
            [
                {
                    "Scenario": scenario.scenario_id,
                    "Demand nodes": len(scenario.nodes) - 1,
                    "Helicopters": len(scenario.helicopters),
                    "Tasks": len(scenario.tasks),
                    "Horizon": scenario.horizon,
                    "Travel CV": uncertainty.travel_time_cv,
                    "Failure probability": uncertainty.failure_probability,
                    "Arrival window": uncertainty.dynamic_arrival_window,
                    "Initially visible": uncertainty.initial_task_fraction,
                    "Training steps": TRAIN_TIMESTEPS,
                    "Evaluation episodes": EVAL_EPISODES,
                }
            ]
        )
        config
        """
    ),
    markdown(
        r"""
        ## 3. Entrenamiento de PPO dentro del notebook

        La clase `NotebookTrainingCallback` guarda la recompensa de cada episodio
        mientras PPO aprende. Es la información principal que antes se observaba en
        TensorBoard, pero ahora queda disponible directamente como un `DataFrame`.
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
                'Faltan las dependencias RL. En la terminal ejecuta: '
                'python -m pip install -e ".[all]"'
            ) from exc


        class NotebookTrainingCallback(BaseCallback):
            # Record episode return and length without TensorBoard.

            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.records: list[dict[str, float]] = []

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for info, done in zip(infos, dones):
                    episode = info.get("episode")
                    if done and episode is not None:
                        self.records.append(
                            {
                                "timestep": int(self.num_timesteps),
                                "episode_return": float(episode["r"]),
                                "episode_length": int(episode["l"]),
                            }
                        )
                return True


        def mask_fn(environment):
            return environment.action_masks()
        """
    ),
    code(
        r"""
        model_dir.mkdir(parents=True, exist_ok=True)
        training_callback = NotebookTrainingCallback()

        if RETRAIN_PPO:
            base_environment = make_environment(SCENARIO_NAME, STOCHASTIC)
            training_environment = Monitor(
                ActionMasker(base_environment, mask_fn)
            )
            model = MaskablePPO(
                "MlpPolicy",
                training_environment,
                seed=TRAIN_SEED,
                verbose=0,
            )
            model.learn(
                total_timesteps=TRAIN_TIMESTEPS,
                callback=training_callback,
                progress_bar=True,
            )
            model.save(model_stem)
            training_environment.close()
            print("Modelo guardado en:", model_path)
        else:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"No existe {model_path}. Activa RETRAIN_PPO = True."
                )
            model = MaskablePPO.load(model_path)
            print("Modelo cargado desde:", model_path)

        training_history = pd.DataFrame(training_callback.records)
        print("Episodios de entrenamiento registrados:", len(training_history))
        """
    ),
    code(
        r"""
        if training_history.empty:
            print(
                "No hay curva nueva porque se cargó un modelo existente. "
                "Usa RETRAIN_PPO = True para observar el aprendizaje."
            )
        else:
            window = min(SMOOTHING_WINDOW, len(training_history))
            training_history["moving_average"] = (
                training_history["episode_return"]
                .rolling(window=window, min_periods=1)
                .mean()
            )

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(
                training_history["timestep"],
                training_history["episode_return"],
                color="#4C78A8",
                alpha=0.18,
                linewidth=1,
                label="Retorno de cada episodio",
            )
            ax.plot(
                training_history["timestep"],
                training_history["moving_average"],
                color="#1F4E79",
                linewidth=2.4,
                label=f"Promedio móvil ({window} episodios)",
            )
            ax.set(
                title="Aprendizaje de PPO",
                xlabel="Pasos de entrenamiento",
                ylabel="Recompensa normalizada del episodio",
            )
            ax.grid(alpha=0.25)
            ax.legend()
            plt.tight_layout()
            plt.show()
        """
    ),
    markdown(
        r"""
        ### Cómo leer la curva

        - Una tendencia ascendente indica que PPO está aprendiendo.
        - Una meseta indica que agregar pasos produce mejoras pequeñas.
        - Una caída sostenida o una variación creciente requiere revisar entrenamiento.
        - Esta curva describe el entrenamiento; la comparación válida con Greedy se hace
          en episodios de evaluación separados.
        """
    ),
    markdown(
        r"""
        ## 4. Evaluación común de Greedy y PPO

        Se utilizan semillas `0, 1, ..., EVAL_EPISODES-1` para ambas políticas.
        Las semillas hacen el experimento reproducible y generan las mismas revelaciones
        iniciales. Sin embargo, el simulador actual consume números aleatorios según las
        acciones realizadas; por ello, después de que las políticas toman rutas distintas,
        no se garantiza un flujo exógeno idéntico evento por evento. Esta es una evaluación
        sobre la misma distribución, todavía no un diseño de *common random numbers* estricto.
        """
    ),
    code(
        r"""
        def result_row(policy_name, seed, summary, steps):
            return {
                "policy": policy_name,
                "seed": int(seed),
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


        def run_greedy_episode(seed):
            result, simulator = run_episode(
                scenario,
                GreedyOnlinePolicy(),
                uncertainty=uncertainty,
                seed=seed,
            )
            return (
                result_row(
                    "Greedy online",
                    seed,
                    simulator.summary(),
                    result.steps,
                ),
                simulator,
            )


        def run_ppo_episode(seed):
            environment = make_environment(SCENARIO_NAME, STOCHASTIC)
            observation, _ = environment.reset(seed=seed)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                    action_masks=environment.action_masks(),
                )
                observation, _, terminated, truncated, _ = environment.step(
                    action
                )
                steps += 1
            row = result_row(
                "PPO",
                seed,
                environment.simulator.summary(),
                steps,
            )
            return row, environment
        """
    ),
    code(
        r"""
        records = []
        for evaluation_seed in range(EVAL_EPISODES):
            greedy_row, _ = run_greedy_episode(evaluation_seed)
            ppo_row, ppo_environment = run_ppo_episode(evaluation_seed)
            records.extend([greedy_row, ppo_row])
            ppo_environment.close()

        results = pd.DataFrame(records).sort_values(
            ["seed", "policy"]
        ).reset_index(drop=True)

        assert len(results) == 2 * EVAL_EPISODES
        assert set(results["policy"]) == {"Greedy online", "PPO"}
        assert results["invalid_actions"].sum() == 0

        if SAVE_CSV:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(results_path, index=False)
            print("CSV guardado en:", results_path)

        results.head(8)
        """
    ),
    markdown(
        r"""
        ## 5. Tabla principal

        El **beneficio normalizado** es la métrica principal porque incorpora prioridad y
        oportunidad temporal. Las tareas completadas, el tiempo de respuesta y el vuelo
        ayudan a explicar cómo se obtiene ese beneficio.
        """
    ),
    code(
        r"""
        METRICS = {
            "benefit": "Beneficio",
            "completed_tasks": "Tareas completadas",
            "average_response_time": "Tiempo de respuesta",
            "flight_time": "Tiempo de vuelo",
            "failures": "Fallas",
        }

        summary_rows = []
        for policy_name, policy_data in results.groupby("policy", sort=False):
            row = {"Política": policy_name, "Episodios": len(policy_data)}
            for metric, label in METRICS.items():
                values = policy_data[metric].astype(float)
                mean_value = values.mean()
                sd_value = values.std(ddof=1)
                ci_half = 1.96 * sd_value / np.sqrt(len(values))
                row[f"{label}: media"] = mean_value
                row[f"{label}: DE"] = sd_value
                row[f"{label}: IC95%"] = (
                    f"[{mean_value - ci_half:.4f}, "
                    f"{mean_value + ci_half:.4f}]"
                )
            summary_rows.append(row)

        summary_table = pd.DataFrame(summary_rows)
        summary_table
        """
    ),
    code(
        r"""
        benefit_by_seed = results.pivot(
            index="seed", columns="policy", values="benefit"
        )
        benefit_by_seed["PPO - Greedy"] = (
            benefit_by_seed["PPO"] - benefit_by_seed["Greedy online"]
        )

        greedy_mean = benefit_by_seed["Greedy online"].mean()
        ppo_mean = benefit_by_seed["PPO"].mean()
        relative_change = (
            100.0 * (ppo_mean - greedy_mean) / greedy_mean
            if greedy_mean != 0
            else np.nan
        )
        comparison_readout = pd.DataFrame(
            [
                {
                    "Greedy mean benefit": greedy_mean,
                    "PPO mean benefit": ppo_mean,
                    "Absolute difference": ppo_mean - greedy_mean,
                    "Relative difference (%)": relative_change,
                    "Seeds PPO > Greedy": int(
                        (benefit_by_seed["PPO - Greedy"] > 0).sum()
                    ),
                    "Seeds PPO = Greedy": int(
                        np.isclose(benefit_by_seed["PPO - Greedy"], 0).sum()
                    ),
                    "Seeds PPO < Greedy": int(
                        (benefit_by_seed["PPO - Greedy"] < 0).sum()
                    ),
                }
            ]
        )
        comparison_readout
        """
    ),
    markdown(
        r"""
        ## 6. ¿PPO supera a Greedy?

        El panel izquierdo muestra la distribución del beneficio. El derecho compara cada
        semilla: sobre la diagonal, PPO obtuvo más beneficio; bajo la diagonal, Greedy fue
        mejor.
        """
    ),
    code(
        r"""
        policy_order = ["Greedy online", "PPO"]
        policy_colors = ["#F28E2B", "#4E79A7"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

        benefit_groups = [
            results.loc[results["policy"] == policy, "benefit"].to_numpy()
            for policy in policy_order
        ]
        box = axes[0].boxplot(
            benefit_groups,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        axes[0].set_xticks([1, 2], policy_order)
        for patch, color in zip(box["boxes"], policy_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for position, values, color in zip(
            range(1, 3), benefit_groups, policy_colors
        ):
            jitter = np.random.default_rng(218 + position).normal(
                position, 0.035, size=len(values)
            )
            axes[0].scatter(
                jitter, values, s=14, color=color, alpha=0.30
            )
        axes[0].set(
            title="Distribución del beneficio",
            ylabel="Beneficio normalizado",
        )
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].scatter(
            benefit_by_seed["Greedy online"],
            benefit_by_seed["PPO"],
            color="#4E79A7",
            alpha=0.70,
            edgecolor="white",
            linewidth=0.4,
        )
        lower = min(
            benefit_by_seed["Greedy online"].min(),
            benefit_by_seed["PPO"].min(),
        )
        upper = max(
            benefit_by_seed["Greedy online"].max(),
            benefit_by_seed["PPO"].max(),
        )
        margin = max(0.01, 0.05 * (upper - lower))
        axes[1].plot(
            [lower - margin, upper + margin],
            [lower - margin, upper + margin],
            "--",
            color="black",
            linewidth=1,
            label="Igual desempeño",
        )
        axes[1].set(
            title="Comparación por semilla",
            xlabel="Beneficio Greedy online",
            ylabel="Beneficio PPO",
            xlim=(lower - margin, upper + margin),
            ylim=(lower - margin, upper + margin),
        )
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 7. ¿Cómo obtiene el resultado cada política?

        Cada gráfico muestra media e intervalo de confianza aproximado de 95%.
        En beneficio y tareas completadas, más alto suele ser mejor. En tiempo de
        respuesta y tiempo de vuelo, más bajo suele ser mejor, siempre interpretado junto
        con el beneficio.
        """
    ),
    code(
        r"""
        dashboard_metrics = [
            ("benefit", "Beneficio normalizado", True),
            ("completed_tasks", "Tareas completadas", True),
            ("average_response_time", "Tiempo medio de respuesta", False),
            ("flight_time", "Tiempo total de vuelo", False),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (metric, title, higher_is_better) in zip(
            axes.flat, dashboard_metrics
        ):
            means = []
            errors = []
            for policy in policy_order:
                values = results.loc[
                    results["policy"] == policy, metric
                ].astype(float)
                means.append(values.mean())
                errors.append(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
            ax.bar(
                policy_order,
                means,
                yerr=errors,
                color=policy_colors,
                alpha=0.75,
                capsize=5,
            )
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
            direction = "más alto es mejor" if higher_is_better else "más bajo es mejor"
            ax.set_xlabel(direction)

        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 8. Un episodio explicado

        Para evitar elegir arbitrariamente un caso favorable, se selecciona la semilla cuya
        diferencia `PPO - Greedy` está más cerca de la mediana. Es un episodio representativo
        de la comparación, no necesariamente el mejor ni el peor.
        """
    ),
    code(
        r"""
        median_gap = benefit_by_seed["PPO - Greedy"].median()
        representative_seed = int(
            (benefit_by_seed["PPO - Greedy"] - median_gap).abs().idxmin()
        )

        greedy_episode, greedy_simulator = run_greedy_episode(
            representative_seed
        )
        ppo_episode, ppo_environment = run_ppo_episode(representative_seed)
        ppo_simulator = ppo_environment.simulator

        representative_summary = pd.DataFrame(
            [greedy_episode, ppo_episode]
        )
        print("Semilla representativa:", representative_seed)
        representative_summary[
            [
                "policy",
                "benefit",
                "completed_tasks",
                "average_response_time",
                "flight_time",
                "failures",
            ]
        ]
        """
    ),
    code(
        r"""
        def cumulative_benefit_trace(simulator):
            times = [0.0]
            values = [0.0]
            cumulative = 0.0
            for record in sorted(simulator.log, key=lambda item: item.time):
                if record.event == "task_completed":
                    cumulative += record.reward
                    times.append(record.time)
                    values.append(cumulative)
            return np.asarray(times), np.asarray(values)


        fig, ax = plt.subplots(figsize=(10, 4.5))
        for policy_name, simulator, color in (
            ("Greedy online", greedy_simulator, policy_colors[0]),
            ("PPO", ppo_simulator, policy_colors[1]),
        ):
            times, values = cumulative_benefit_trace(simulator)
            ax.step(
                times,
                values,
                where="post",
                linewidth=2.3,
                color=color,
                label=policy_name,
            )

        ax.set(
            title=f"Beneficio acumulado — semilla {representative_seed}",
            xlabel="Tiempo de misión (min)",
            ylabel="Beneficio normalizado acumulado",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        def operational_intervals(simulator, policy_name):
            task_starts = {}
            return_starts = {}
            failure_starts = {}
            rows = []

            for record in sorted(simulator.log, key=lambda item: item.time):
                if record.event == "task_dispatched":
                    task_starts[(record.helicopter_id, record.task_id)] = record.time
                elif record.event == "task_completed":
                    key = (record.helicopter_id, record.task_id)
                    start = task_starts.pop(key, record.time)
                    priority = simulator.scenario.task_by_id[
                        record.task_id
                    ].priority_class.value
                    rows.append(
                        {
                            "policy": policy_name,
                            "helicopter": record.helicopter_id,
                            "start": start,
                            "end": record.time,
                            "label": record.task_id,
                            "kind": priority,
                        }
                    )
                elif record.event == "return_dispatched":
                    return_starts[record.helicopter_id] = record.time
                elif record.event == "return_completed":
                    start = return_starts.pop(
                        record.helicopter_id, record.time
                    )
                    rows.append(
                        {
                            "policy": policy_name,
                            "helicopter": record.helicopter_id,
                            "start": start,
                            "end": record.time,
                            "label": "BASE",
                            "kind": "return",
                        }
                    )
                elif record.event == "dispatch_failure":
                    failure_starts[record.helicopter_id] = record.time
                elif record.event == "failure_recovered":
                    start = failure_starts.pop(
                        record.helicopter_id, record.time
                    )
                    rows.append(
                        {
                            "policy": policy_name,
                            "helicopter": record.helicopter_id,
                            "start": start,
                            "end": record.time,
                            "label": "FALLA",
                            "kind": "failure",
                        }
                    )
            return pd.DataFrame(rows)


        intervals = pd.concat(
            [
                operational_intervals(
                    greedy_simulator, "Greedy online"
                ),
                operational_intervals(ppo_simulator, "PPO"),
            ],
            ignore_index=True,
        )

        gantt_colors = {
            "medical": "#E15759",
            "personnel": "#59A14F",
            "cargo": "#4E79A7",
            "return": "#BAB0AC",
            "failure": "#B07AA1",
        }
        helicopter_order = [
            helicopter.helicopter_id for helicopter in scenario.helicopters
        ]

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 3.1 * 2), sharex=True
        )
        for ax, policy_name in zip(axes, policy_order):
            policy_intervals = intervals.loc[
                intervals["policy"] == policy_name
            ]
            for _, interval in policy_intervals.iterrows():
                y = helicopter_order.index(interval["helicopter"])
                width = interval["end"] - interval["start"]
                ax.barh(
                    y,
                    width,
                    left=interval["start"],
                    height=0.62,
                    color=gantt_colors[interval["kind"]],
                    edgecolor="white",
                    linewidth=0.6,
                )
                if width >= 2.0:
                    ax.text(
                        interval["start"] + width / 2,
                        y,
                        interval["label"],
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white"
                        if interval["kind"] not in ("return",)
                        else "black",
                    )
            ax.set_yticks(range(len(helicopter_order)))
            ax.set_yticklabels(helicopter_order)
            ax.invert_yaxis()
            ax.set_title(policy_name)
            ax.set_ylabel("Helicóptero")
            ax.grid(axis="x", alpha=0.2)

        axes[-1].set_xlabel("Tiempo de misión (min)")
        legend_handles = [
            plt.Rectangle(
                (0, 0), 1, 1, color=color, label=kind.capitalize()
            )
            for kind, color in gantt_colors.items()
        ]
        axes[0].legend(
            handles=legend_handles,
            ncol=len(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.33),
        )
        fig.suptitle(
            f"Cronograma operacional — semilla {representative_seed}",
            y=1.03,
        )
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        def task_outcomes(simulator, policy_name):
            reward_by_task = {
                record.task_id: record.reward
                for record in simulator.log
                if record.event == "task_completed"
            }
            rows = []
            for task in simulator.scenario.tasks:
                state = simulator.task_states[task.task_id]
                rows.append(
                    {
                        "policy": policy_name,
                        "task": task.task_id,
                        "priority": task.priority_class.value,
                        "release": simulator.release_times[task.task_id],
                        "status": state.status.value,
                        "completion": state.completion_time,
                        "response": (
                            state.completion_time
                            - simulator.release_times[task.task_id]
                            if state.completion_time is not None
                            else np.nan
                        ),
                        "benefit": reward_by_task.get(task.task_id, 0.0),
                    }
                )
            return pd.DataFrame(rows)


        task_table = pd.concat(
            [
                task_outcomes(greedy_simulator, "Greedy online"),
                task_outcomes(ppo_simulator, "PPO"),
            ],
            ignore_index=True,
        ).sort_values(["task", "policy"])

        task_table
        """
    ),
    code(
        r"""
        ppo_environment.close()

        print("Validación final")
        print("----------------")
        print("Políticas comparadas:", sorted(results["policy"].unique()))
        print("Episodios por política:", EVAL_EPISODES)
        print("Acciones inválidas:", int(results["invalid_actions"].sum()))
        print("Modelo:", model_path)
        if SAVE_CSV:
            print("Resultados:", results_path)
        """
    ),
    markdown(
        r"""
        ## 9. Siguiente progresión recomendada

        Ejecuta y conserva los resultados por etapas:

        1. **Comprensión:** `small`, 20 000 pasos, una semilla de entrenamiento.
        2. **Estabilidad:** `small`, 100 000 pasos, comparar la nueva curva y evaluación.
        3. **Variabilidad del aprendizaje:** repetir entrenamiento con varias semillas.
        4. **Escenario de interés:** cambiar a `talcahuano`, volver a entrenar y evaluar.
        5. **Diseño científico:** separar los flujos aleatorios exógenos para que Greedy y
           PPO enfrenten exactamente las mismas realizaciones evento por evento.

        Los CSV quedan como respaldo y análisis posterior. La ejecución, los parámetros,
        la curva de aprendizaje, las métricas y la interpretación principal permanecen
        dentro del notebook.
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"ttap-gp-{index:02d}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (Python 3.11)",
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
    / "01_greedy_online_vs_ppo.ipynb"
)
destination.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(destination)
