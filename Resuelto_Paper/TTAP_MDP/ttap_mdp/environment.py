"""Optional Gymnasium adapter for PPO and DQN."""

from __future__ import annotations

from typing import Any

from .dynamics import Simulator, UncertaintyConfig
from .entities import HelicopterStatus, TaskStatus
from .scenario import Scenario


try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError:  # Core simulator intentionally has no third-party dependency.
    gym = None
    np = None
    spaces = None


_H_STATUS = {
    HelicopterStatus.AVAILABLE: 0.0,
    HelicopterStatus.BUSY: 0.5,
    HelicopterStatus.UNAVAILABLE: 1.0,
}
_T_STATUS = {
    TaskStatus.UNREVEALED: 0.0,
    TaskStatus.PENDING: 0.25,
    TaskStatus.ASSIGNED: 0.5,
    TaskStatus.COMPLETED: 0.75,
    TaskStatus.EXPIRED: 1.0,
}


class _MissingGymBase:
    pass


_GymBase = gym.Env if gym is not None else _MissingGymBase


class TTAPGymEnv(_GymBase):
    """Fixed-size normalized observation and discrete masked action space."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        scenario: Scenario,
        uncertainty: UncertaintyConfig | None = None,
        *,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        if gym is None:
            raise ImportError(
                "Gymnasium is required. Install with: pip install -e '.[rl]'"
            )
        super().__init__()
        self.scenario = scenario
        self.uncertainty = uncertainty or UncertaintyConfig.deterministic()
        self.invalid_action_penalty = invalid_action_penalty
        self.simulator = Simulator(
            scenario,
            self.uncertainty,
            strict_actions=False,
            invalid_action_penalty=invalid_action_penalty,
        )
        self.action_space = spaces.Discrete(len(self.simulator.actions))
        observation_length = 2 + 7 * len(scenario.helicopters) + 9 * len(scenario.tasks)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_length,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        episode_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )
        self.simulator.reset(seed=episode_seed)
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.simulator.step(int(action))
        return (
            self._observation(),
            float(result.reward),
            result.terminated,
            False,
            self._info(),
        )

    def action_masks(self) -> Any:
        """Interface expected by sb3-contrib MaskablePPO."""

        return np.asarray(self.simulator.action_mask(), dtype=bool)

    def _observation(self) -> Any:
        scenario = self.scenario
        node_ids = [node.node_id for node in scenario.nodes]
        node_denominator = max(1, len(node_ids) - 1)
        horizon = scenario.horizon
        values: list[float] = [
            self.simulator.time / horizon,
            min(1.0, self.simulator.cumulative_reward),
        ]

        for helicopter in scenario.helicopters:
            state = self.simulator.helicopter_states[helicopter.helicopter_id]
            capacity = helicopter.capacity
            values.extend(
                [
                    node_ids.index(state.node_id) / node_denominator,
                    state.remaining_resources.cargo / max(1.0, capacity.cargo),
                    state.remaining_resources.medical / max(1.0, capacity.medical),
                    state.remaining_resources.personnel
                    / max(1.0, capacity.personnel),
                    _H_STATUS[state.status],
                    max(0.0, state.available_at - self.simulator.time) / horizon,
                    float(helicopter.visual_capable),
                ]
            )

        max_capacity = (
            max(h.capacity.cargo for h in scenario.helicopters),
            max(h.capacity.medical for h in scenario.helicopters),
            max(h.capacity.personnel for h in scenario.helicopters),
        )
        max_weight = max(task.priority_weight for task in scenario.tasks)
        for task in scenario.tasks:
            state = self.simulator.task_states[task.task_id]
            values.extend(
                [
                    _T_STATUS[state.status],
                    node_ids.index(task.node_id) / node_denominator,
                    task.requirements.cargo / max(1.0, max_capacity[0]),
                    task.requirements.medical / max(1.0, max_capacity[1]),
                    task.requirements.personnel / max(1.0, max_capacity[2]),
                    task.optimal_time / horizon,
                    task.effective_time / horizon,
                    task.ineffective_time / horizon,
                    task.priority_weight / max_weight,
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            **self.simulator.summary(),
            "action_mask": self.action_masks(),
        }

    def render(self) -> str:
        summary = self.simulator.summary()
        return (
            f"t={summary['time']:.1f} | benefit={summary['benefit']:.4f} | "
            f"completed={summary['completed_tasks']}/{len(self.scenario.tasks)}"
        )
