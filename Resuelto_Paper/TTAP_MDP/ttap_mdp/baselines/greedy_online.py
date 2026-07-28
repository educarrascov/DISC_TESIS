"""Immediate-benefit online Greedy baseline."""

from __future__ import annotations

from ..actions import ActionKind
from ..dynamics import Simulator
from ..rewards import normalized_task_benefit


class GreedyOnlinePolicy:
    """Reproduce the paper's asynchronous online Greedy decision rule.

    Helicopters are considered in fleet order among those available at the
    current event time. For the first one with a feasible decision, the policy
    chooses its first task with maximum immediate benefit using the paper's
    node/task order. If it cannot execute another task away from base, it
    returns and recovers.
    """

    name = "Greedy online"

    def reset(self, seed: int | None = None) -> None:
        del seed

    def select_action(self, simulator: Simulator) -> int:
        mask = simulator.action_mask()
        node_order = {
            node.node_id: position
            for position, node in enumerate(simulator.scenario.nodes)
        }
        task_order = {
            task.task_id: (
                node_order[task.node_id],
                position,
            )
            for position, task in enumerate(simulator.scenario.tasks)
        }
        for helicopter in simulator.scenario.helicopters:
            helicopter_id = helicopter.helicopter_id
            candidates: list[tuple[float, tuple[int, int], int]] = []
            for index, (action, feasible) in enumerate(
                zip(simulator.actions, mask)
            ):
                if (
                    not feasible
                    or action.kind is not ActionKind.ASSIGN
                    or action.helicopter_id != helicopter_id
                ):
                    continue
                completion = simulator.nominal_completion_time(action)
                task = simulator.scenario.task_by_id[action.task_id]
                benefit = normalized_task_benefit(
                    simulator.scenario,
                    task,
                    completion,
                )
                candidates.append((-benefit, task_order[action.task_id], index))
            if candidates:
                return min(candidates)[-1]

            return_index = simulator.action_index(
                ActionKind.RETURN,
                helicopter_id,
            )
            if mask[return_index]:
                return return_index
        return 0
