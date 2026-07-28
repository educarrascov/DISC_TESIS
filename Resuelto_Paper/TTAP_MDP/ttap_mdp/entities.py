"""Core entities for the deterministic TTAP model.

This module intentionally contains no routing, transition, reward, or
reinforcement-learning logic.  Its only responsibility is to define validated
data structures shared by the future simulator and optimization baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


_TOLERANCE = 1e-9


class EntityValidationError(ValueError):
    """Raised when an entity violates a basic TTAP data invariant."""


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EntityValidationError(f"{field_name} must be a nonempty string.")


def _validate_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise EntityValidationError(f"{field_name} must be finite.")


def _validate_nonnegative(value: float, field_name: str) -> None:
    _validate_finite(value, field_name)
    if value < 0:
        raise EntityValidationError(f"{field_name} must be nonnegative.")


@dataclass(frozen=True, slots=True)
class ResourceVector:
    """Cargo, medical, and personnel resource quantities."""

    cargo: float = 0.0
    medical: float = 0.0
    personnel: float = 0.0

    def __post_init__(self) -> None:
        _validate_nonnegative(self.cargo, "cargo")
        _validate_nonnegative(self.medical, "medical")
        _validate_nonnegative(self.personnel, "personnel")

    @property
    def is_zero(self) -> bool:
        return (
            abs(self.cargo) <= _TOLERANCE
            and abs(self.medical) <= _TOLERANCE
            and abs(self.personnel) <= _TOLERANCE
        )

    def can_cover(self, requirement: "ResourceVector") -> bool:
        """Return whether this vector contains every required resource."""

        return (
            self.cargo + _TOLERANCE >= requirement.cargo
            and self.medical + _TOLERANCE >= requirement.medical
            and self.personnel + _TOLERANCE >= requirement.personnel
        )

    def consume(self, requirement: "ResourceVector") -> "ResourceVector":
        """Return the remaining resources after a feasible consumption."""

        if not self.can_cover(requirement):
            raise EntityValidationError(
                "The resource requirement exceeds the available resources."
            )

        return ResourceVector(
            cargo=max(0.0, self.cargo - requirement.cargo),
            medical=max(0.0, self.medical - requirement.medical),
            personnel=max(0.0, self.personnel - requirement.personnel),
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return self.cargo, self.medical, self.personnel


class CoordinateSystem(str, Enum):
    """Coordinate representation used by a node."""

    CARTESIAN_KM = "cartesian_km"
    GEOGRAPHIC_DEGREES = "geographic_degrees"


@dataclass(frozen=True, slots=True)
class Node:
    """A base or demand location."""

    node_id: str
    first_coordinate: float
    second_coordinate: float
    coordinate_system: CoordinateSystem = CoordinateSystem.CARTESIAN_KM
    is_base: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.node_id, "node_id")
        _validate_finite(self.first_coordinate, "first_coordinate")
        _validate_finite(self.second_coordinate, "second_coordinate")

        if self.name is not None:
            _validate_identifier(self.name, "name")

        if self.coordinate_system is CoordinateSystem.GEOGRAPHIC_DEGREES:
            if not -90.0 <= self.first_coordinate <= 90.0:
                raise EntityValidationError(
                    "A geographic latitude must be between -90 and 90 degrees."
                )
            if not -180.0 <= self.second_coordinate <= 180.0:
                raise EntityValidationError(
                    "A geographic longitude must be between -180 and 180 degrees."
                )

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.first_coordinate, self.second_coordinate


class TaskPriority(str, Enum):
    """Operational priority class used in the first TTAP study."""

    MEDICAL = "medical"
    PERSONNEL = "personnel"
    CARGO = "cargo"


@dataclass(frozen=True, slots=True)
class Task:
    """Immutable definition of a task."""

    task_id: str
    node_id: str
    requirements: ResourceVector
    optimal_time: float
    effective_time: float
    ineffective_time: float
    priority_weight: float
    service_time: float
    release_time: float = 0.0
    requires_visual_capability: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.task_id, "task_id")
        _validate_identifier(self.node_id, "node_id")
        _validate_nonnegative(self.release_time, "release_time")
        _validate_nonnegative(self.optimal_time, "optimal_time")
        _validate_nonnegative(self.effective_time, "effective_time")
        _validate_nonnegative(self.ineffective_time, "ineffective_time")
        _validate_nonnegative(self.service_time, "service_time")
        _validate_finite(self.priority_weight, "priority_weight")

        if self.requirements.is_zero:
            raise EntityValidationError(
                "A task must require at least one operational resource."
            )
        if not self.optimal_time < self.effective_time < self.ineffective_time:
            raise EntityValidationError(
                "Task thresholds must satisfy "
                "optimal_time < effective_time < ineffective_time."
            )
        if self.priority_weight <= 0:
            raise EntityValidationError("priority_weight must be strictly positive.")

    @property
    def priority_class(self) -> TaskPriority:
        """Classify a task using medical > personnel > cargo precedence."""

        if self.requirements.medical > _TOLERANCE:
            return TaskPriority.MEDICAL
        if self.requirements.personnel > _TOLERANCE:
            return TaskPriority.PERSONNEL
        return TaskPriority.CARGO


@dataclass(frozen=True, slots=True)
class Helicopter:
    """Immutable definition of a helicopter."""

    helicopter_id: str
    model: str
    initial_node_id: str
    capacity: ResourceVector
    speed_kmh: float
    visual_capable: bool

    def __post_init__(self) -> None:
        _validate_identifier(self.helicopter_id, "helicopter_id")
        _validate_identifier(self.model, "model")
        _validate_identifier(self.initial_node_id, "initial_node_id")
        _validate_finite(self.speed_kmh, "speed_kmh")

        if self.speed_kmh <= 0:
            raise EntityValidationError("speed_kmh must be strictly positive.")

    def is_compatible_with(self, task: Task) -> bool:
        """Return static resource and visual-capability compatibility."""

        visual_compatible = (
            not task.requires_visual_capability or self.visual_capable
        )
        return visual_compatible and self.capacity.can_cover(task.requirements)


class HelicopterStatus(str, Enum):
    """Operational state of a helicopter during a simulation."""

    AVAILABLE = "available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class HelicopterState:
    """Time-varying state associated with one helicopter."""

    helicopter_id: str
    node_id: str
    remaining_resources: ResourceVector
    status: HelicopterStatus = HelicopterStatus.AVAILABLE
    available_at: float = 0.0
    active_task_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.helicopter_id, "helicopter_id")
        _validate_identifier(self.node_id, "node_id")
        _validate_nonnegative(self.available_at, "available_at")

        if self.active_task_id is not None:
            _validate_identifier(self.active_task_id, "active_task_id")

        if (
            self.status is HelicopterStatus.AVAILABLE
            and self.active_task_id is not None
        ):
            raise EntityValidationError(
                "An available helicopter cannot have an active task."
            )


class TaskStatus(str, Enum):
    """Lifecycle state of a task during a simulation."""

    UNREVEALED = "unrevealed"
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class TaskState:
    """Time-varying state associated with one task."""

    task_id: str
    status: TaskStatus
    assigned_helicopter_id: str | None = None
    completion_time: float | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.task_id, "task_id")

        if self.assigned_helicopter_id is not None:
            _validate_identifier(
                self.assigned_helicopter_id,
                "assigned_helicopter_id",
            )
        if self.completion_time is not None:
            _validate_nonnegative(self.completion_time, "completion_time")

        if (
            self.status is TaskStatus.ASSIGNED
            and self.assigned_helicopter_id is None
        ):
            raise EntityValidationError(
                "An assigned task must identify its helicopter."
            )
        if self.status is TaskStatus.COMPLETED and self.completion_time is None:
            raise EntityValidationError(
                "A completed task must have a completion time."
            )
        if self.status is not TaskStatus.COMPLETED and self.completion_time is not None:
            raise EntityValidationError(
                "Only a completed task may have a completion time."
            )


def initial_helicopter_state(helicopter: Helicopter) -> HelicopterState:
    """Build the deterministic initial state of a helicopter."""

    return HelicopterState(
        helicopter_id=helicopter.helicopter_id,
        node_id=helicopter.initial_node_id,
        remaining_resources=helicopter.capacity,
    )


def initial_task_state(task: Task, current_time: float = 0.0) -> TaskState:
    """Build the initial task state from its deterministic release time."""

    _validate_nonnegative(current_time, "current_time")
    status = (
        TaskStatus.PENDING
        if task.release_time <= current_time + _TOLERANCE
        else TaskStatus.UNREVEALED
    )
    return TaskState(task_id=task.task_id, status=status)
