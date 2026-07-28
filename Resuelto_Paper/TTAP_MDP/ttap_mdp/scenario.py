"""Scenario definitions and validated TTAP benchmark builders."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

from .entities import (
    CoordinateSystem,
    Helicopter,
    Node,
    ResourceVector,
    Task,
)


class ScenarioValidationError(ValueError):
    """Raised when a scenario violates a structural TTAP invariant."""


@dataclass(frozen=True, slots=True)
class Scenario:
    """Immutable collection of nodes, helicopters, tasks, and time parameters."""

    scenario_id: str
    nodes: tuple[Node, ...]
    helicopters: tuple[Helicopter, ...]
    tasks: tuple[Task, ...]
    horizon: float = 200.0
    recovery_time: float = 2.0
    alpha: float = 0.1
    time_unit: float = 1.0

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ScenarioValidationError("scenario_id must be nonempty.")
        if self.horizon <= 0:
            raise ScenarioValidationError("horizon must be strictly positive.")
        if self.recovery_time < 0:
            raise ScenarioValidationError("recovery_time must be nonnegative.")
        if self.alpha <= 0:
            raise ScenarioValidationError("alpha must be strictly positive.")
        if self.time_unit <= 0:
            raise ScenarioValidationError("time_unit must be strictly positive.")
        if not self.nodes or not self.helicopters or not self.tasks:
            raise ScenarioValidationError(
                "A scenario requires nodes, helicopters, and tasks."
            )

        self._validate_unique_ids()
        base_nodes = [node for node in self.nodes if node.is_base]
        if len(base_nodes) != 1:
            raise ScenarioValidationError(
                "A scenario must contain exactly one base node."
            )

        coordinate_systems = {node.coordinate_system for node in self.nodes}
        if len(coordinate_systems) != 1:
            raise ScenarioValidationError(
                "All nodes must use the same coordinate system."
            )

        node_ids = set(self.node_by_id)
        for helicopter in self.helicopters:
            if helicopter.initial_node_id not in node_ids:
                raise ScenarioValidationError(
                    f"{helicopter.helicopter_id} references an unknown initial node."
                )
        for task in self.tasks:
            if task.node_id not in node_ids:
                raise ScenarioValidationError(
                    f"{task.task_id} references an unknown demand node."
                )
            if task.release_time > self.horizon:
                raise ScenarioValidationError(
                    f"{task.task_id} is released after the scenario horizon."
                )
            if not any(h.is_compatible_with(task) for h in self.helicopters):
                raise ScenarioValidationError(
                    f"{task.task_id} has no statically compatible helicopter."
                )

    def _validate_unique_ids(self) -> None:
        for label, identifiers in (
            ("node", [item.node_id for item in self.nodes]),
            ("helicopter", [item.helicopter_id for item in self.helicopters]),
            ("task", [item.task_id for item in self.tasks]),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ScenarioValidationError(f"Duplicate {label} identifier.")

    @property
    def node_by_id(self) -> dict[str, Node]:
        return {node.node_id: node for node in self.nodes}

    @property
    def helicopter_by_id(self) -> dict[str, Helicopter]:
        return {helicopter.helicopter_id: helicopter for helicopter in self.helicopters}

    @property
    def task_by_id(self) -> dict[str, Task]:
        return {task.task_id: task for task in self.tasks}

    @property
    def base(self) -> Node:
        return next(node for node in self.nodes if node.is_base)

    @property
    def total_task_weight(self) -> float:
        return sum(task.priority_weight for task in self.tasks)

    def distance_km(self, first_node_id: str, second_node_id: str) -> float:
        """Return Cartesian or Haversine distance in kilometres."""

        try:
            first = self.node_by_id[first_node_id]
            second = self.node_by_id[second_node_id]
        except KeyError as exc:
            raise ScenarioValidationError(f"Unknown node: {exc.args[0]}") from exc

        if first.coordinate_system is CoordinateSystem.CARTESIAN_KM:
            dx = first.first_coordinate - second.first_coordinate
            dy = first.second_coordinate - second.second_coordinate
            return sqrt(dx * dx + dy * dy)

        lat1, lon1 = map(radians, first.coordinates)
        lat2, lon2 = map(radians, second.coordinates)
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        haversine = (
            sin(delta_lat / 2.0) ** 2
            + cos(lat1) * cos(lat2) * sin(delta_lon / 2.0) ** 2
        )
        return 2.0 * 6371.0088 * asin(sqrt(haversine))

    def nominal_travel_time(
        self,
        helicopter_id: str,
        first_node_id: str,
        second_node_id: str,
    ) -> float:
        helicopter = self.helicopter_by_id[helicopter_id]
        return 60.0 * self.distance_km(first_node_id, second_node_id) / helicopter.speed_kmh


def build_small_scenario() -> Scenario:
    """Return a fast six-task scenario for tests, examples, and RL smoke runs."""

    nodes = (
        Node("BASE", 0.0, 0.0, is_base=True, name="Operations base"),
        Node("A", 8.0, 0.0, name="Demand A"),
        Node("B", 4.0, 6.0, name="Demand B"),
        Node("C", 10.0, 8.0, name="Demand C"),
    )
    helicopters = (
        Helicopter(
            "H1",
            "Light_visual",
            "BASE",
            ResourceVector(cargo=300, medical=1, personnel=2),
            speed_kmh=180,
            visual_capable=True,
        ),
        Helicopter(
            "H2",
            "Medium_nonvisual",
            "BASE",
            ResourceVector(cargo=500, medical=2, personnel=4),
            speed_kmh=200,
            visual_capable=False,
        ),
    )
    task_rows = (
        ("T1", "A", 100, 1, 0, 8, 25, 60, 2.0, True),
        ("T2", "B", 0, 0, 2, 12, 35, 75, 1.5, False),
        ("T3", "C", 250, 0, 0, 18, 45, 90, 1.0, False),
        ("T4", "A", 100, 0, 1, 22, 50, 100, 1.5, False),
        ("T5", "B", 150, 1, 0, 25, 55, 110, 2.0, True),
        ("T6", "C", 300, 0, 0, 30, 65, 125, 1.0, False),
    )
    tasks = tuple(
        Task(
            task_id=task_id,
            node_id=node_id,
            requirements=ResourceVector(cargo, medical, personnel),
            optimal_time=optimal,
            effective_time=effective,
            ineffective_time=ineffective,
            priority_weight=weight,
            service_time=6.0,
            requires_visual_capability=visual,
        )
        for (
            task_id,
            node_id,
            cargo,
            medical,
            personnel,
            optimal,
            effective,
            ineffective,
            weight,
            visual,
        ) in task_rows
    )
    return Scenario(
        scenario_id="SMALL_3areas_2helos_6tasks",
        nodes=nodes,
        helicopters=helicopters,
        tasks=tasks,
        horizon=140.0,
        recovery_time=2.0,
        alpha=0.1,
        time_unit=1.0,
    )


_GPS_NODES = (
    ("BASE", "Talcahuano Base", -36.69111351421891, -73.0478599920742, True),
    ("A", "Cancha Colegio Arturo Prat", -36.70062541801277, -73.12141380985622, False),
    ("B", "Estacionamiento Hospital Higueras", -36.738871104828384, -73.1102591903089, False),
    ("C", "Estadio CAP Huachipato", -36.75513285670206, -73.10765044444817, False),
    ("D", "Club Hipico", -36.78339332263817, -73.09259088926588, False),
    ("E", "Kingston College", -36.784954466865976, -73.05085478019805, False),
    ("F", "Parque Bicentenario", -36.83173804040988, -73.06322287960043, False),
    ("G", "Estadio Ester Roa", -36.81499794336835, -73.02359653839595, False),
)

_BASE_TASK_ROWS = (
    (1, True, 200, 2, 1, 20, 40, 60),
    (2, False, 300, 1, 2, 15, 30, 50),
    (3, True, 100, 3, 0, 10, 25, 40),
    (4, False, 200, 0, 3, 30, 50, 70),
    (5, True, 150, 1, 1, 25, 45, 60),
    (6, True, 100, 3, 0, 10, 25, 120),
    (7, False, 400, 0, 3, 30, 50, 90),
    (8, True, 150, 1, 1, 25, 45, 100),
    (9, True, 200, 2, 1, 20, 40, 60),
    (10, False, 300, 1, 2, 15, 30, 50),
    (11, True, 100, 3, 0, 10, 25, 40),
    (12, False, 200, 0, 3, 30, 50, 70),
    (13, True, 150, 1, 1, 25, 45, 60),
    (14, True, 100, 3, 0, 10, 25, 120),
    (15, False, 400, 0, 3, 30, 50, 90),
    (16, True, 150, 1, 1, 25, 45, 100),
    (17, True, 150, 1, 1, 25, 45, 60),
    (18, True, 100, 3, 0, 10, 25, 120),
    (19, False, 400, 0, 3, 30, 50, 90),
    (20, True, 150, 1, 1, 25, 45, 100),
)

_BASE_TASK_NODES = {
    1: "A", 2: "A", 9: "A", 10: "A",
    3: "B", 7: "B", 8: "B", 11: "B", 12: "B", 13: "B", 14: "B",
    5: "C", 6: "C", 15: "C", 16: "C",
    4: "D", 17: "D", 18: "D", 19: "D", 20: "D",
}


def build_talcahuano_scenario() -> Scenario:
    """Return the validated 7-area, 6-helicopter, 30-task paper scenario."""

    nodes = tuple(
        Node(
            node_id=node_id,
            name=name,
            first_coordinate=latitude,
            second_coordinate=longitude,
            coordinate_system=CoordinateSystem.GEOGRAPHIC_DEGREES,
            is_base=is_base,
        )
        for node_id, name, latitude, longitude, is_base in _GPS_NODES
    )
    fleet_rows = (
        ("H1", "H125_light", True, 300, 1, 2, 190),
        ("H2", "H125_light", False, 300, 1, 2, 190),
        ("H3", "AS365_Dauphin", True, 500, 2, 4, 210),
        ("H4", "AS365_Dauphin", False, 500, 2, 4, 210),
        ("H5", "AS532_Cougar", True, 900, 3, 8, 200),
        ("H6", "AS532_Cougar", False, 900, 3, 8, 200),
    )
    helicopters = tuple(
        Helicopter(
            helicopter_id,
            model,
            "BASE",
            ResourceVector(cargo, medical, personnel),
            speed,
            visual,
        )
        for helicopter_id, model, visual, cargo, medical, personnel, speed in fleet_rows
    )

    rows = list(_BASE_TASK_ROWS)
    for new_id, row in zip(range(21, 31), _BASE_TASK_ROWS[:10]):
        _, visual, cargo, medical, personnel, optimal, effective, ineffective = row
        rows.append(
            (new_id, visual, cargo, medical, personnel, optimal, effective, ineffective)
        )

    new_nodes = ("E", "F", "G")
    tasks: list[Task] = []
    for row in rows:
        task_id, visual, cargo, medical, personnel, optimal, effective, ineffective = row
        node_id = (
            _BASE_TASK_NODES[task_id]
            if task_id <= 20
            else new_nodes[(task_id - 21) % len(new_nodes)]
        )
        weight = 2.0 if medical > 0 else 1.5 if personnel > 0 else 1.0
        tasks.append(
            Task(
                task_id=f"T{task_id}",
                node_id=node_id,
                requirements=ResourceVector(cargo, medical, personnel),
                optimal_time=optimal,
                effective_time=effective,
                ineffective_time=ineffective,
                priority_weight=weight,
                service_time=6.0,
                requires_visual_capability=visual,
            )
        )

    return Scenario(
        scenario_id="GPS_7areas_6helos_30tasks",
        nodes=nodes,
        helicopters=helicopters,
        tasks=tuple(tasks),
        horizon=200.0,
        recovery_time=2.0,
        alpha=0.1,
        time_unit=1.0,
    )
