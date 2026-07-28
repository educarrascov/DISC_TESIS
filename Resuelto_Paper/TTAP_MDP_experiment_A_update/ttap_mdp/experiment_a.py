"""Experiment-A scenario family used by the deterministic TTAP study.

The geometry, fleet, task-profile generator, and spatial-allocation streams in
this module mirror Experiment A.  For a fixed instance seed, the 20-, 30-, 40-,
50-, and 60-task scenarios are nested prefixes of the same 60-task instance.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import ceil, floor

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - covered by project dependency.
    raise ImportError(
        "Experiment A requires NumPy. Install with: pip install -e '.[all]'"
    ) from exc

from .entities import Helicopter, Node, ResourceVector, Task
from .scenario import Scenario


EXPERIMENT_A_TASK_COUNTS = (20, 30, 40, 50, 60)
EXPERIMENT_A_INSTANCE_SEEDS = tuple(range(1, 31))
EXPERIMENT_A_MAX_TASKS = max(EXPERIMENT_A_TASK_COUNTS)
EXPERIMENT_A_GENERATOR_VERSION = "matched_v2_heterogeneous"

_TASK_PROFILE_STREAM_ID = 21001
_TASK_LOCATION_STREAM_ID = 22001
_TASK_CLASS_NAMES = ("medical", "personnel", "cargo")
_TASK_CLASS_PROBABILITIES = (0.45, 0.30, 0.25)
_VISUAL_REQUIRED_PROBABILITY = 0.40

_NODES = (
    ("BASE", 0.0, 0.0, True),
    ("A", 10.0, 5.0, False),
    ("B", 20.0, 15.0, False),
    ("C", 30.0, 10.0, False),
    ("D", 40.0, 20.0, False),
)

_FLEET = (
    ("H1", True, 500, 3, 3, 150),
    ("H2", False, 500, 3, 3, 120),
    ("H3", True, 500, 3, 3, 140),
    ("H4", False, 500, 3, 3, 160),
)


def _rng(seed: int, stream_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream_id]))


def _priority_class(profile: dict[str, float | int | bool]) -> str:
    if int(profile["medical"]) > 0:
        return "medical"
    if int(profile["personnel"]) > 0:
        return "personnel"
    return "cargo"


def _priority_weight(profile: dict[str, float | int | bool]) -> float:
    return {"medical": 2.0, "personnel": 1.5, "cargo": 1.0}[
        _priority_class(profile)
    ]


def _draw_task_profile(
    rng: np.random.Generator,
) -> dict[str, float | int | bool]:
    task_class = str(rng.choice(_TASK_CLASS_NAMES, p=_TASK_CLASS_PROBABILITIES))
    visual = bool(rng.random() < _VISUAL_REQUIRED_PROBABILITY)

    if task_class == "medical":
        cargo = int(rng.choice(np.arange(50, 501, 50)))
        medical = int(rng.integers(1, 4))
        personnel = int(rng.integers(0, 4))
    elif task_class == "personnel":
        cargo = int(rng.choice(np.arange(50, 501, 50)))
        medical = 0
        personnel = int(rng.integers(1, 4))
    else:
        cargo = int(rng.choice(np.arange(100, 501, 50)))
        medical = 0
        personnel = 0

    optimal = int(rng.integers(8, 36))
    effective = optimal + int(rng.integers(12, 27))
    ineffective = effective + int(rng.integers(20, 61))
    profile: dict[str, float | int | bool] = {
        "visual": visual,
        "cargo": cargo,
        "medical": medical,
        "personnel": personnel,
        "optimal": float(optimal),
        "effective": float(effective),
        "ineffective": float(ineffective),
    }
    if _priority_class(profile) != task_class:
        raise AssertionError("Generated priority class is inconsistent.")
    return profile


def _prefix_is_complex(
    profiles: dict[int, dict[str, float | int | bool]],
    n_tasks: int,
) -> bool:
    prefix = [profiles[index] for index in range(1, n_tasks + 1)]
    class_counts = Counter(_priority_class(profile) for profile in prefix)
    minimum_class_count = max(3, int(floor(0.12 * n_tasks)))
    if any(class_counts[name] < minimum_class_count for name in _TASK_CLASS_NAMES):
        return False

    visual_count = sum(bool(profile["visual"]) for profile in prefix)
    if not (
        ceil(0.25 * n_tasks)
        <= visual_count
        <= floor(0.60 * n_tasks)
    ):
        return False

    resource_vectors = {
        (
            int(profile["cargo"]),
            int(profile["medical"]),
            int(profile["personnel"]),
        )
        for profile in prefix
    }
    return len(resource_vectors) >= ceil(0.65 * n_tasks)


@lru_cache(maxsize=None)
def generate_experiment_a_task_profiles(
    n_tasks: int,
    instance_seed: int,
) -> tuple[dict[str, float | int | bool], ...]:
    """Generate the nested task-profile prefix for one Experiment-A instance."""

    if n_tasks not in EXPERIMENT_A_TASK_COUNTS:
        raise ValueError(
            f"n_tasks must belong to {EXPERIMENT_A_TASK_COUNTS}; got {n_tasks}."
        )
    rng = _rng(instance_seed, _TASK_PROFILE_STREAM_ID)
    for _ in range(10_000):
        candidate = {
            task_id: _draw_task_profile(rng)
            for task_id in range(1, EXPERIMENT_A_MAX_TASKS + 1)
        }
        if all(
            _prefix_is_complex(candidate, prefix_size)
            for prefix_size in EXPERIMENT_A_TASK_COUNTS
        ):
            return tuple(
                candidate[task_id].copy()
                for task_id in range(1, n_tasks + 1)
            )
    raise RuntimeError("Could not generate a sufficiently complex task set.")


@lru_cache(maxsize=None)
def generate_experiment_a_locations(
    n_tasks: int,
    instance_seed: int,
) -> tuple[str, ...]:
    """Generate the asymmetric, nested four-area task allocation."""

    if n_tasks not in EXPERIMENT_A_TASK_COUNTS:
        raise ValueError(
            f"n_tasks must belong to {EXPERIMENT_A_TASK_COUNTS}; got {n_tasks}."
        )
    node_choices = ("A", "B", "C", "D")
    probabilities = np.asarray((0.45, 0.30, 0.17, 0.08))
    rng = _rng(instance_seed, _TASK_LOCATION_STREAM_ID)

    for _ in range(10_000):
        hotspot_order = list(rng.permutation(node_choices))
        sequence = tuple(
            str(node)
            for node in rng.choice(
                hotspot_order,
                size=EXPERIMENT_A_MAX_TASKS,
                replace=True,
                p=probabilities,
            )
        )
        valid = True
        for prefix_size in EXPERIMENT_A_TASK_COUNTS:
            counts = Counter(sequence[:prefix_size])
            values = np.asarray([counts[node] for node in node_choices])
            required_imbalance = max(4, ceil(0.20 * prefix_size))
            if np.any(values < 1) or values.max() - values.min() < required_imbalance:
                valid = False
                break
        if valid:
            return sequence[:n_tasks]
    raise RuntimeError("Could not generate an asymmetric spatial allocation.")


@lru_cache(maxsize=None)
def build_experiment_a_scenario(
    n_tasks: int = 20,
    instance_seed: int = 1,
) -> Scenario:
    """Build one exact Experiment-A geometry/fleet/task-profile instance."""

    profiles = generate_experiment_a_task_profiles(n_tasks, instance_seed)
    locations = generate_experiment_a_locations(n_tasks, instance_seed)
    nodes = tuple(
        Node(
            node_id=node_id,
            first_coordinate=x,
            second_coordinate=y,
            is_base=is_base,
            name="Operations base" if is_base else f"Demand area {node_id}",
        )
        for node_id, x, y, is_base in _NODES
    )
    helicopters = tuple(
        Helicopter(
            helicopter_id=helicopter_id,
            model="Experiment_A",
            initial_node_id="BASE",
            capacity=ResourceVector(cargo, medical, personnel),
            speed_kmh=speed,
            visual_capable=visual,
        )
        for (
            helicopter_id,
            visual,
            cargo,
            medical,
            personnel,
            speed,
        ) in _FLEET
    )
    tasks = tuple(
        Task(
            task_id=f"T{index}",
            node_id=locations[index - 1],
            requirements=ResourceVector(
                cargo=float(profile["cargo"]),
                medical=float(profile["medical"]),
                personnel=float(profile["personnel"]),
            ),
            optimal_time=float(profile["optimal"]),
            effective_time=float(profile["effective"]),
            ineffective_time=float(profile["ineffective"]),
            priority_weight=_priority_weight(profile),
            service_time=6.0,
            requires_visual_capability=bool(profile["visual"]),
        )
        for index, profile in enumerate(profiles, start=1)
    )
    return Scenario(
        scenario_id=(
            f"EXP_A_4areas_4helos_{n_tasks}tasks_seed{int(instance_seed)}"
        ),
        nodes=nodes,
        helicopters=helicopters,
        tasks=tasks,
        horizon=200.0,
        recovery_time=2.0,
        alpha=0.1,
        time_unit=1.0,
    )

