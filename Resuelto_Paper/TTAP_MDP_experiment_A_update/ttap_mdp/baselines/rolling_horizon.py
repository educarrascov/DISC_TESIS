"""Rolling-horizon exact matching policy."""

from __future__ import annotations

from functools import lru_cache

from ..actions import ActionKind
from ..dynamics import Simulator
from ..rewards import normalized_task_benefit


class RollingHorizonPolicy:
    """Solve an exact one-dispatch-per-helicopter assignment at each epoch.

    The epoch subproblem is the same binary matching problem commonly written
    as a small MILP. It is solved here by dynamic programming, avoiding a solver
    dependency while preserving the exact assignment optimum for that epoch.
    It is a rolling-horizon comparator, not the full state-arc MILP from Paper 1.
    """

    name = "Rolling-horizon matching"

    def reset(self, seed: int | None = None) -> None:
        del seed

    def select_action(self, simulator: Simulator) -> int:
        mask = simulator.action_mask()
        helicopter_ids = [
            helicopter.helicopter_id
            for helicopter in simulator.scenario.helicopters
        ]
        helicopter_index = {
            helicopter_id: index
            for index, helicopter_id in enumerate(helicopter_ids)
        }
        tasks = [task.task_id for task in simulator.scenario.tasks]
        candidates: dict[str, list[tuple[int, int, float]]] = {
            task_id: [] for task_id in tasks
        }
        action_values: dict[int, float] = {}
        for action_index, (action, feasible) in enumerate(
            zip(simulator.actions, mask)
        ):
            if not feasible or action.kind is not ActionKind.ASSIGN:
                continue
            completion = simulator.nominal_completion_time(action)
            value = normalized_task_benefit(
                simulator.scenario,
                simulator.scenario.task_by_id[action.task_id],
                completion,
            )
            h_index = helicopter_index[action.helicopter_id]
            candidates[action.task_id].append((h_index, action_index, value))
            action_values[action_index] = value

        @lru_cache(maxsize=None)
        def solve(task_position: int, used_helicopters: int) -> tuple[float, tuple[int, ...]]:
            if task_position >= len(tasks):
                return 0.0, ()
            task_id = tasks[task_position]
            best_value, best_actions = solve(task_position + 1, used_helicopters)
            for h_index, action_index, value in candidates[task_id]:
                bit = 1 << h_index
                if used_helicopters & bit:
                    continue
                future_value, future_actions = solve(
                    task_position + 1,
                    used_helicopters | bit,
                )
                option_value = value + future_value
                if option_value > best_value + 1e-12:
                    best_value = option_value
                    best_actions = (action_index,) + future_actions
            return best_value, best_actions

        _, plan = solve(0, 0)
        if plan:
            return min(
                plan,
                key=lambda index: (
                    -action_values[index],
                    simulator.nominal_completion_time(simulator.actions[index]),
                    simulator.actions[index].label,
                ),
            )

        returns = [
            index
            for index, (action, feasible) in enumerate(zip(simulator.actions, mask))
            if feasible and action.kind is ActionKind.RETURN
        ]
        return returns[0] if returns else 0
