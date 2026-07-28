"""TTAP-MDP v1.0 research code."""

from .actions import Action, ActionKind, build_action_catalog
from .action_mask import feasible_action_labels, feasible_action_mask
from .dynamics import (
    EventRecord,
    InvalidActionError,
    Simulator,
    StepResult,
    UncertaintyConfig,
)
from .entities import (
    CoordinateSystem,
    EntityValidationError,
    Helicopter,
    HelicopterState,
    HelicopterStatus,
    Node,
    ResourceVector,
    Task,
    TaskPriority,
    TaskState,
    TaskStatus,
    initial_helicopter_state,
    initial_task_state,
)
from .evaluation import (
    EpisodeResult,
    aggregate_results,
    evaluate_policy,
    format_results_table,
    run_episode,
)
from .experiment_a import (
    EXPERIMENT_A_GENERATOR_VERSION,
    EXPERIMENT_A_INSTANCE_SEEDS,
    EXPERIMENT_A_MAX_TASKS,
    EXPERIMENT_A_TASK_COUNTS,
    build_experiment_a_scenario,
    generate_experiment_a_locations,
    generate_experiment_a_task_profiles,
)
from .rewards import normalized_task_benefit, temporal_satisfaction
from .environment import ExperimentAFamilyEnv, TTAPGymEnv
from .scenario import (
    Scenario,
    ScenarioValidationError,
    build_small_scenario,
    build_talcahuano_scenario,
)

__all__ = [
    "Action",
    "ActionKind",
    "CoordinateSystem",
    "EntityValidationError",
    "EpisodeResult",
    "EXPERIMENT_A_GENERATOR_VERSION",
    "EXPERIMENT_A_INSTANCE_SEEDS",
    "EXPERIMENT_A_MAX_TASKS",
    "EXPERIMENT_A_TASK_COUNTS",
    "ExperimentAFamilyEnv",
    "EventRecord",
    "Helicopter",
    "HelicopterState",
    "HelicopterStatus",
    "Node",
    "ResourceVector",
    "Scenario",
    "ScenarioValidationError",
    "Simulator",
    "StepResult",
    "Task",
    "TaskPriority",
    "TaskState",
    "TaskStatus",
    "TTAPGymEnv",
    "UncertaintyConfig",
    "InvalidActionError",
    "aggregate_results",
    "build_action_catalog",
    "build_experiment_a_scenario",
    "build_small_scenario",
    "build_talcahuano_scenario",
    "evaluate_policy",
    "feasible_action_labels",
    "feasible_action_mask",
    "format_results_table",
    "generate_experiment_a_locations",
    "generate_experiment_a_task_profiles",
    "initial_helicopter_state",
    "initial_task_state",
    "normalized_task_benefit",
    "run_episode",
    "temporal_satisfaction",
]
