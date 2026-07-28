"""Public action-mask helpers."""

from __future__ import annotations

from .dynamics import Simulator


def feasible_action_mask(simulator: Simulator) -> tuple[bool, ...]:
    """Return a Boolean mask aligned with ``simulator.actions``."""

    return simulator.action_mask()


def feasible_action_labels(simulator: Simulator) -> tuple[str, ...]:
    """Return human-readable labels for every currently feasible action."""

    return tuple(
        action.label
        for action, allowed in zip(simulator.actions, simulator.action_mask())
        if allowed
    )
