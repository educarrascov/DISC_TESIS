"""Event-driven deterministic and stochastic TTAP simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import heapq
from math import ceil, exp, log, sqrt
import random
from typing import Any

from .actions import Action, ActionKind, build_action_catalog
from .entities import (
    HelicopterState,
    HelicopterStatus,
    TaskState,
    TaskStatus,
    initial_helicopter_state,
)
from .rewards import normalized_task_benefit, temporal_satisfaction
from .scenario import Scenario


_TOLERANCE = 1e-9


class InvalidActionError(ValueError):
    """Raised when strict simulation receives an infeasible action."""


@dataclass(frozen=True, slots=True)
class UncertaintyConfig:
    """Episode-level stochastic dynamics.

    ``travel_time_cv`` uses a mean-one lognormal multiplier. A failure is an
    operational unavailability event sampled immediately before task dispatch.
    Dynamic arrivals preserve task definitions and sample only reveal times.
    """

    travel_time_cv: float = 0.0
    failure_probability: float = 0.0
    downtime_min: float = 5.0
    downtime_max: float = 20.0
    dynamic_arrival_window: float = 0.0
    initial_task_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.travel_time_cv < 0:
            raise ValueError("travel_time_cv must be nonnegative.")
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be in [0, 1].")
        if self.downtime_min < 0 or self.downtime_max < self.downtime_min:
            raise ValueError("Invalid downtime interval.")
        if self.dynamic_arrival_window < 0:
            raise ValueError("dynamic_arrival_window must be nonnegative.")
        if not 0.0 <= self.initial_task_fraction <= 1.0:
            raise ValueError("initial_task_fraction must be in [0, 1].")

    @classmethod
    def deterministic(cls) -> "UncertaintyConfig":
        return cls()

    @classmethod
    def moderate(cls) -> "UncertaintyConfig":
        return cls(
            travel_time_cv=0.15,
            failure_probability=0.05,
            downtime_min=5.0,
            downtime_max=15.0,
            dynamic_arrival_window=45.0,
            initial_task_fraction=0.35,
        )


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    time: float
    kind: str
    helicopter_id: str
    task_id: str | None = None
    destination_node_id: str | None = None
    remaining_resources: Any = None
    flight_time: float = 0.0
    start_time: float = 0.0


@dataclass(frozen=True, slots=True)
class EventRecord:
    time: float
    event: str
    helicopter_id: str | None = None
    task_id: str | None = None
    node_id: str | None = None
    flight_time: float = 0.0
    reward: float = 0.0


@dataclass(frozen=True, slots=True)
class StepResult:
    reward: float
    terminated: bool
    action: Action
    valid_action: bool
    info: dict[str, Any]


class Simulator:
    """Centralized finite-horizon TTAP event simulator."""

    def __init__(
        self,
        scenario: Scenario,
        uncertainty: UncertaintyConfig | None = None,
        *,
        seed: int = 0,
        strict_actions: bool = False,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        self.scenario = scenario
        self.uncertainty = uncertainty or UncertaintyConfig.deterministic()
        self.strict_actions = strict_actions
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.actions = build_action_catalog(scenario)
        self._action_to_index = {action: index for index, action in enumerate(self.actions)}
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        elif not hasattr(self, "seed"):
            self.seed = 0
        self.rng = random.Random(self.seed)
        self.time = 0.0
        self.cumulative_reward = 0.0
        self.total_flight_time = 0.0
        self.failures = 0
        self.invalid_actions = 0
        self._counter = 0
        self._assignment_attempts: dict[tuple[str, str], int] = {}
        self._return_attempts: dict[tuple[str, str], int] = {}
        self._events: list[tuple[float, int, SimulationEvent]] = []
        self.log: list[EventRecord] = []
        self.helicopter_states = {
            helicopter.helicopter_id: initial_helicopter_state(helicopter)
            for helicopter in self.scenario.helicopters
        }
        self.release_times = self._sample_release_times()
        self.task_states = {
            task.task_id: TaskState(
                task_id=task.task_id,
                status=(
                    TaskStatus.PENDING
                    if self.release_times[task.task_id] <= _TOLERANCE
                    else TaskStatus.UNREVEALED
                ),
            )
            for task in self.scenario.tasks
        }
        self.terminated = False
        self._refresh_terminal_state()
        return self.snapshot()

    def _sample_release_times(self) -> dict[str, float]:
        releases: dict[str, float] = {}
        config = self.uncertainty
        for task in self.scenario.tasks:
            release = task.release_time
            event_rng = self._keyed_rng("release", task.task_id)
            if (
                config.dynamic_arrival_window > 0
                and event_rng.random() > config.initial_task_fraction
            ):
                release = max(
                    release,
                    event_rng.uniform(0.0, config.dynamic_arrival_window),
                )
            releases[task.task_id] = self._quantize_absolute(release)
        return releases

    def _keyed_rng(self, *parts: object) -> random.Random:
        """Return a counter-based stream independent of policy action order."""

        payload = "|".join((str(self.seed), *(str(part) for part in parts)))
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:16], "big"))

    def _quantize_duration(self, duration: float) -> float:
        unit = self.scenario.time_unit
        return ceil((duration - _TOLERANCE) / unit) * unit

    def _quantize_absolute(self, time_value: float) -> float:
        return self._quantize_duration(max(0.0, time_value))

    def _travel_multiplier(self, event_rng: random.Random) -> float:
        cv = self.uncertainty.travel_time_cv
        if cv <= 0:
            return 1.0
        sigma_squared = log(1.0 + cv * cv)
        sigma = sqrt(sigma_squared)
        mean = -0.5 * sigma_squared
        return exp(event_rng.normalvariate(mean, sigma))

    def _sample_travel_time(
        self,
        helicopter_id: str,
        first_node_id: str,
        second_node_id: str,
        event_rng: random.Random,
    ) -> float:
        nominal = self.scenario.nominal_travel_time(
            helicopter_id,
            first_node_id,
            second_node_id,
        )
        return nominal * self._travel_multiplier(event_rng)

    def nominal_completion_time(self, action: Action) -> float:
        if action.kind is not ActionKind.ASSIGN:
            return self.time
        state = self.helicopter_states[action.helicopter_id]
        task = self.scenario.task_by_id[action.task_id]
        travel = self.scenario.nominal_travel_time(
            action.helicopter_id,
            state.node_id,
            task.node_id,
        )
        return self.time + self._quantize_duration(travel + task.service_time)

    def action_index(
        self,
        kind: ActionKind,
        helicopter_id: str | None = None,
        task_id: str | None = None,
    ) -> int:
        return self._action_to_index[Action(kind, helicopter_id, task_id)]

    def action_mask(self) -> tuple[bool, ...]:
        return tuple(self.is_action_feasible(action) for action in self.actions)

    def is_action_feasible(self, action: Action) -> bool:
        if self.terminated:
            return False
        if action.kind is ActionKind.WAIT:
            return True

        state = self.helicopter_states[action.helicopter_id]
        if state.status is not HelicopterStatus.AVAILABLE:
            return False

        if action.kind is ActionKind.RETURN:
            if state.node_id == self.scenario.base.node_id:
                return False
            travel = self.scenario.nominal_travel_time(
                action.helicopter_id,
                state.node_id,
                self.scenario.base.node_id,
            )
            finish = self.time + self._quantize_duration(
                travel + self.scenario.recovery_time
            )
            return finish <= self.scenario.horizon + _TOLERANCE

        task_state = self.task_states[action.task_id]
        if task_state.status is not TaskStatus.PENDING:
            return False
        task = self.scenario.task_by_id[action.task_id]
        helicopter = self.scenario.helicopter_by_id[action.helicopter_id]
        if task.requires_visual_capability and not helicopter.visual_capable:
            return False
        if not state.remaining_resources.can_cover(task.requirements):
            return False

        completion = self.nominal_completion_time(action)
        return_travel = self.scenario.nominal_travel_time(
            action.helicopter_id,
            task.node_id,
            self.scenario.base.node_id,
        )
        recovered = completion + self._quantize_duration(
            return_travel + self.scenario.recovery_time
        )
        return (
            recovered <= self.scenario.horizon + _TOLERANCE
            and normalized_task_benefit(self.scenario, task, completion) > 0.0
        )

    def step(self, action_index: int) -> StepResult:
        if self.terminated:
            raise RuntimeError("Cannot step a terminated simulation.")
        if not 0 <= int(action_index) < len(self.actions):
            raise IndexError("action_index is outside the action catalogue.")

        action = self.actions[int(action_index)]
        valid = self.is_action_feasible(action)
        if not valid and self.strict_actions:
            raise InvalidActionError(f"Infeasible action: {action.label}")
        if not valid:
            self.invalid_actions += 1
            self.log.append(EventRecord(self.time, "invalid_action"))
            reward = self.invalid_action_penalty + self._advance_time()
            self.cumulative_reward += self.invalid_action_penalty
            return self._step_result(reward, action, False)

        reward = 0.0
        if action.kind is ActionKind.WAIT:
            reward = self._advance_time()
        elif action.kind is ActionKind.RETURN:
            self._dispatch_return(action)
        else:
            self._dispatch_assignment(action)

        if action.kind is not ActionKind.WAIT and not any(self.action_mask()[1:]):
            reward += self._advance_time()
        self._refresh_terminal_state()
        return self._step_result(reward, action, True)

    def _step_result(self, reward: float, action: Action, valid: bool) -> StepResult:
        return StepResult(
            reward=reward,
            terminated=self.terminated,
            action=action,
            valid_action=valid,
            info=self.summary(),
        )

    def _push_event(self, event: SimulationEvent) -> None:
        self._counter += 1
        heapq.heappush(self._events, (event.time, self._counter, event))

    def _dispatch_assignment(self, action: Action) -> None:
        helicopter_id = action.helicopter_id
        task_id = action.task_id
        state = self.helicopter_states[helicopter_id]
        attempt_key = (helicopter_id, task_id)
        attempt = self._assignment_attempts.get(attempt_key, 0)
        self._assignment_attempts[attempt_key] = attempt + 1
        event_rng = self._keyed_rng(
            "assignment",
            helicopter_id,
            task_id,
            attempt,
        )

        if event_rng.random() < self.uncertainty.failure_probability:
            downtime = event_rng.uniform(
                self.uncertainty.downtime_min,
                self.uncertainty.downtime_max,
            )
            available_at = min(
                self.scenario.horizon,
                self.time + self._quantize_duration(downtime),
            )
            self.helicopter_states[helicopter_id] = replace(
                state,
                status=HelicopterStatus.UNAVAILABLE,
                available_at=available_at,
                active_task_id=None,
            )
            self.failures += 1
            self.log.append(
                EventRecord(
                    self.time,
                    "dispatch_failure",
                    helicopter_id=helicopter_id,
                    node_id=state.node_id,
                )
            )
            self._push_event(
                SimulationEvent(
                    available_at,
                    "failure_recovery",
                    helicopter_id,
                    destination_node_id=state.node_id,
                )
            )
            return

        task = self.scenario.task_by_id[task_id]
        flight_time = self._sample_travel_time(
            helicopter_id,
            state.node_id,
            task.node_id,
            event_rng,
        )
        completion_time = self.time + self._quantize_duration(
            flight_time + task.service_time
        )
        remaining = state.remaining_resources.consume(task.requirements)
        self.helicopter_states[helicopter_id] = replace(
            state,
            status=HelicopterStatus.BUSY,
            available_at=completion_time,
            active_task_id=task_id,
        )
        self.task_states[task_id] = TaskState(
            task_id,
            TaskStatus.ASSIGNED,
            assigned_helicopter_id=helicopter_id,
        )
        self.total_flight_time += flight_time
        self.log.append(
            EventRecord(
                self.time,
                "task_dispatched",
                helicopter_id,
                task_id,
                task.node_id,
                flight_time,
            )
        )
        self._push_event(
            SimulationEvent(
                completion_time,
                "task_completion",
                helicopter_id,
                task_id=task_id,
                destination_node_id=task.node_id,
                remaining_resources=remaining,
                flight_time=flight_time,
                start_time=self.time,
            )
        )

    def _dispatch_return(self, action: Action) -> None:
        helicopter_id = action.helicopter_id
        state = self.helicopter_states[helicopter_id]
        attempt_key = (helicopter_id, state.node_id)
        attempt = self._return_attempts.get(attempt_key, 0)
        self._return_attempts[attempt_key] = attempt + 1
        event_rng = self._keyed_rng(
            "return",
            helicopter_id,
            state.node_id,
            attempt,
        )
        flight_time = self._sample_travel_time(
            helicopter_id,
            state.node_id,
            self.scenario.base.node_id,
            event_rng,
        )
        completion_time = self.time + self._quantize_duration(
            flight_time + self.scenario.recovery_time
        )
        self.helicopter_states[helicopter_id] = replace(
            state,
            status=HelicopterStatus.BUSY,
            available_at=completion_time,
            active_task_id=None,
        )
        self.total_flight_time += flight_time
        self.log.append(
            EventRecord(
                self.time,
                "return_dispatched",
                helicopter_id,
                node_id=self.scenario.base.node_id,
                flight_time=flight_time,
            )
        )
        self._push_event(
            SimulationEvent(
                completion_time,
                "return_completion",
                helicopter_id,
                destination_node_id=self.scenario.base.node_id,
                flight_time=flight_time,
                start_time=self.time,
            )
        )

    def _advance_time(self) -> float:
        next_event_time = self._events[0][0] if self._events else float("inf")
        unrevealed = [
            self.release_times[task_id]
            for task_id, state in self.task_states.items()
            if state.status is TaskStatus.UNREVEALED
            and self.release_times[task_id] > self.time + _TOLERANCE
        ]
        next_release_time = min(unrevealed) if unrevealed else float("inf")
        target = min(next_event_time, next_release_time, self.scenario.horizon)
        if target == float("inf") or target <= self.time + _TOLERANCE:
            target = self.scenario.horizon

        self.time = target
        reward = self._process_current_time()
        self._reveal_and_expire_tasks()
        if self.time >= self.scenario.horizon - _TOLERANCE:
            self.time = self.scenario.horizon
        self._refresh_terminal_state()
        return reward

    def _process_current_time(self) -> float:
        reward = 0.0
        while self._events and self._events[0][0] <= self.time + _TOLERANCE:
            _, _, event = heapq.heappop(self._events)
            state = self.helicopter_states[event.helicopter_id]
            if event.kind == "task_completion":
                task = self.scenario.task_by_id[event.task_id]
                task_reward = normalized_task_benefit(self.scenario, task, event.time)
                self.task_states[event.task_id] = TaskState(
                    event.task_id,
                    TaskStatus.COMPLETED,
                    assigned_helicopter_id=event.helicopter_id,
                    completion_time=event.time,
                )
                self.helicopter_states[event.helicopter_id] = replace(
                    state,
                    node_id=event.destination_node_id,
                    remaining_resources=event.remaining_resources,
                    status=HelicopterStatus.AVAILABLE,
                    available_at=event.time,
                    active_task_id=None,
                )
                reward += task_reward
                self.log.append(
                    EventRecord(
                        event.time,
                        "task_completed",
                        event.helicopter_id,
                        event.task_id,
                        event.destination_node_id,
                        event.flight_time,
                        task_reward,
                    )
                )
            elif event.kind == "return_completion":
                helicopter = self.scenario.helicopter_by_id[event.helicopter_id]
                self.helicopter_states[event.helicopter_id] = replace(
                    state,
                    node_id=self.scenario.base.node_id,
                    remaining_resources=helicopter.capacity,
                    status=HelicopterStatus.AVAILABLE,
                    available_at=event.time,
                    active_task_id=None,
                )
                self.log.append(
                    EventRecord(
                        event.time,
                        "return_completed",
                        event.helicopter_id,
                        node_id=self.scenario.base.node_id,
                    )
                )
            else:
                self.helicopter_states[event.helicopter_id] = replace(
                    state,
                    status=HelicopterStatus.AVAILABLE,
                    available_at=event.time,
                    active_task_id=None,
                )
                self.log.append(
                    EventRecord(
                        event.time,
                        "failure_recovered",
                        event.helicopter_id,
                        node_id=state.node_id,
                    )
                )
        self.cumulative_reward += reward
        return reward

    def _reveal_and_expire_tasks(self) -> None:
        for task in self.scenario.tasks:
            state = self.task_states[task.task_id]
            if (
                state.status is TaskStatus.UNREVEALED
                and self.release_times[task.task_id] <= self.time + _TOLERANCE
            ):
                self.task_states[task.task_id] = TaskState(
                    task.task_id,
                    TaskStatus.PENDING,
                )
                self.log.append(
                    EventRecord(
                        self.time,
                        "task_revealed",
                        task_id=task.task_id,
                        node_id=task.node_id,
                    )
                )
                state = self.task_states[task.task_id]
            if (
                state.status in (TaskStatus.PENDING, TaskStatus.UNREVEALED)
                and self.time >= task.ineffective_time - _TOLERANCE
            ):
                self.task_states[task.task_id] = TaskState(
                    task.task_id,
                    TaskStatus.EXPIRED,
                )

    def _refresh_terminal_state(self) -> None:
        if self.time >= self.scenario.horizon - _TOLERANCE:
            for task_id, state in tuple(self.task_states.items()):
                if state.status not in (TaskStatus.COMPLETED, TaskStatus.EXPIRED):
                    self.task_states[task_id] = TaskState(
                        task_id,
                        TaskStatus.EXPIRED,
                    )
            self.terminated = True
            return

        tasks_finished = all(
            state.status in (TaskStatus.COMPLETED, TaskStatus.EXPIRED)
            for state in self.task_states.values()
        )
        helicopters_recovered = all(
            state.status is HelicopterStatus.AVAILABLE
            and state.node_id == self.scenario.base.node_id
            and state.remaining_resources
            == self.scenario.helicopter_by_id[helicopter_id].capacity
            for helicopter_id, state in self.helicopter_states.items()
        )
        self.terminated = tasks_finished and helicopters_recovered

    def completed_task_ids(self) -> tuple[str, ...]:
        return tuple(
            task.task_id
            for task in self.scenario.tasks
            if self.task_states[task.task_id].status is TaskStatus.COMPLETED
        )

    def summary(self) -> dict[str, Any]:
        completed = self.completed_task_ids()
        completion_times = [
            self.task_states[task_id].completion_time for task_id in completed
        ]
        response_times = [
            self.task_states[task_id].completion_time - self.release_times[task_id]
            for task_id in completed
        ]
        relevant = sum(
            temporal_satisfaction(
                self.task_states[task_id].completion_time,
                self.scenario.task_by_id[task_id],
                self.scenario.alpha,
            )
            >= 0.5
            for task_id in completed
        )
        operational_times = [
            record.time
            for record in self.log
            if record.event
            in ("task_completed", "return_completed", "failure_recovered")
        ]
        return {
            "scenario": self.scenario.scenario_id,
            "time": self.time,
            "benefit": self.cumulative_reward,
            "completed_tasks": len(completed),
            "completion_rate": len(completed) / len(self.scenario.tasks),
            "relevant_completion_rate": relevant / len(self.scenario.tasks),
            "average_response_time": (
                sum(response_times) / len(response_times) if response_times else 0.0
            ),
            "makespan": max(operational_times) if operational_times else 0.0,
            "flight_time": self.total_flight_time,
            "failures": self.failures,
            "invalid_actions": self.invalid_actions,
            "terminated": self.terminated,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "helicopters": dict(self.helicopter_states),
            "tasks": dict(self.task_states),
            "release_times": dict(self.release_times),
            "action_mask": self.action_mask(),
            "summary": self.summary(),
        }
