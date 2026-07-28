"""Fixed discrete action catalogue for the centralized TTAP controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .scenario import Scenario


class ActionKind(str, Enum):
    WAIT = "wait"
    RETURN = "return"
    ASSIGN = "assign"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    helicopter_id: str | None = None
    task_id: str | None = None

    @property
    def label(self) -> str:
        if self.kind is ActionKind.WAIT:
            return "WAIT"
        if self.kind is ActionKind.RETURN:
            return f"RETURN({self.helicopter_id})"
        return f"ASSIGN({self.helicopter_id},{self.task_id})"


def build_action_catalog(scenario: Scenario) -> tuple[Action, ...]:
    """Build WAIT, one return per helicopter, and every assignment pair."""

    actions: list[Action] = [Action(ActionKind.WAIT)]
    actions.extend(
        Action(ActionKind.RETURN, helicopter_id=helicopter.helicopter_id)
        for helicopter in scenario.helicopters
    )
    actions.extend(
        Action(
            ActionKind.ASSIGN,
            helicopter_id=helicopter.helicopter_id,
            task_id=task.task_id,
        )
        for helicopter in scenario.helicopters
        for task in scenario.tasks
    )
    return tuple(actions)
