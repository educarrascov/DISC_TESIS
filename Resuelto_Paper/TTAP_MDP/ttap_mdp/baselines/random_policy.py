"""Uniform random feasible baseline."""

from __future__ import annotations

import random

from ..dynamics import Simulator


class RandomFeasiblePolicy:
    name = "Random feasible"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def select_action(self, simulator: Simulator) -> int:
        feasible = [
            index
            for index, allowed in enumerate(simulator.action_mask())
            if allowed
        ]
        non_wait = [index for index in feasible if index != 0]
        return self.rng.choice(non_wait or feasible)
