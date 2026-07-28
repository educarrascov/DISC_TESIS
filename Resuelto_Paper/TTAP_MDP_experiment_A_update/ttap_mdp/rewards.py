"""Temporal satisfaction and normalized TTAP reward functions."""

from __future__ import annotations

from math import exp

from .entities import Task
from .scenario import Scenario


def temporal_satisfaction(completion_time: float, task: Task, alpha: float = 0.1) -> float:
    """Return the anchored piecewise-logistic satisfaction in ``[0, 1]``.

    The function equals 1 at and before the optimal threshold, 0.5 at the
    effective threshold, and 0 at and after the ineffective threshold.
    """

    if completion_time <= task.optimal_time:
        return 1.0
    if completion_time >= task.ineffective_time:
        return 0.0

    def logistic(value: float) -> float:
        return 1.0 / (1.0 + exp(alpha * (value - task.effective_time)))

    value = logistic(completion_time)
    if completion_time <= task.effective_time:
        at_optimal = logistic(task.optimal_time)
        result = 0.5 + 0.5 * (value - 0.5) / (at_optimal - 0.5)
    else:
        at_ineffective = logistic(task.ineffective_time)
        result = 0.5 * (value - at_ineffective) / (0.5 - at_ineffective)
    return min(1.0, max(0.0, result))


def normalized_task_benefit(
    scenario: Scenario,
    task: Task,
    completion_time: float,
) -> float:
    """Return the priority-weighted task contribution to total reward."""

    return (
        task.priority_weight
        * temporal_satisfaction(completion_time, task, scenario.alpha)
        / scenario.total_task_weight
    )
