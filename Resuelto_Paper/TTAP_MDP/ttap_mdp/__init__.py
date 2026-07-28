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
from .rewards import normalized_task_benefit, temporal_satisfaction
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
    "UncertaintyConfig",
    "InvalidActionError",
    "aggregate_results",
    "build_action_catalog",
    "build_small_scenario",
    "build_talcahuano_scenario",
    "evaluate_policy",
    "feasible_action_labels",
    "feasible_action_mask",
    "format_results_table",
    "initial_helicopter_state",
    "initial_task_state",
    "normalized_task_benefit",
    "run_episode",
    "temporal_satisfaction",
]
