"""Unit tests for the TTAP entity definitions."""

import unittest

from ttap_mdp import (
    CoordinateSystem,
    EntityValidationError,
    Helicopter,
    Node,
    ResourceVector,
    Task,
    TaskPriority,
    TaskStatus,
    initial_helicopter_state,
    initial_task_state,
)


class ResourceVectorTests(unittest.TestCase):
    def test_feasible_consumption_returns_remaining_resources(self) -> None:
        capacity = ResourceVector(cargo=300, medical=1, personnel=2)
        requirement = ResourceVector(cargo=100, medical=1, personnel=0)

        remaining = capacity.consume(requirement)

        self.assertEqual(remaining, ResourceVector(cargo=200, medical=0, personnel=2))

    def test_infeasible_consumption_is_rejected(self) -> None:
        capacity = ResourceVector(cargo=50, medical=0, personnel=1)
        requirement = ResourceVector(cargo=100, medical=0, personnel=0)

        with self.assertRaises(EntityValidationError):
            capacity.consume(requirement)

    def test_negative_resources_are_rejected(self) -> None:
        with self.assertRaises(EntityValidationError):
            ResourceVector(cargo=-1)


class NodeTests(unittest.TestCase):
    def test_valid_geographic_node(self) -> None:
        node = Node(
            node_id="BASE",
            name="Talcahuano Base",
            first_coordinate=-36.6911135,
            second_coordinate=-73.04786,
            coordinate_system=CoordinateSystem.GEOGRAPHIC_DEGREES,
            is_base=True,
        )

        self.assertTrue(node.is_base)
        self.assertEqual(node.coordinates, (-36.6911135, -73.04786))

    def test_invalid_latitude_is_rejected(self) -> None:
        with self.assertRaises(EntityValidationError):
            Node(
                node_id="X",
                first_coordinate=-100.0,
                second_coordinate=-73.0,
                coordinate_system=CoordinateSystem.GEOGRAPHIC_DEGREES,
            )


class TaskAndHelicopterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helicopter = Helicopter(
            helicopter_id="H1",
            model="H125_light",
            initial_node_id="BASE",
            capacity=ResourceVector(cargo=300, medical=1, personnel=2),
            speed_kmh=190,
            visual_capable=True,
        )

    def test_medical_priority_has_precedence(self) -> None:
        task = Task(
            task_id="T1",
            node_id="A",
            requirements=ResourceVector(cargo=100, medical=1, personnel=1),
            optimal_time=10,
            effective_time=30,
            ineffective_time=90,
            priority_weight=2.0,
            service_time=6,
        )

        self.assertEqual(task.priority_class, TaskPriority.MEDICAL)

    def test_static_compatibility_checks_resources_and_visual_capability(self) -> None:
        feasible_task = Task(
            task_id="T1",
            node_id="A",
            requirements=ResourceVector(cargo=100),
            optimal_time=10,
            effective_time=30,
            ineffective_time=90,
            priority_weight=1.0,
            service_time=6,
            requires_visual_capability=True,
        )
        excessive_task = Task(
            task_id="T2",
            node_id="A",
            requirements=ResourceVector(cargo=400),
            optimal_time=10,
            effective_time=30,
            ineffective_time=90,
            priority_weight=1.0,
            service_time=6,
        )

        self.assertTrue(self.helicopter.is_compatible_with(feasible_task))
        self.assertFalse(self.helicopter.is_compatible_with(excessive_task))

    def test_invalid_temporal_thresholds_are_rejected(self) -> None:
        with self.assertRaises(EntityValidationError):
            Task(
                task_id="T1",
                node_id="A",
                requirements=ResourceVector(cargo=100),
                optimal_time=30,
                effective_time=20,
                ineffective_time=90,
                priority_weight=1.0,
                service_time=6,
            )

    def test_initial_states_match_deterministic_assumptions(self) -> None:
        task = Task(
            task_id="T1",
            node_id="A",
            requirements=ResourceVector(cargo=100),
            optimal_time=10,
            effective_time=30,
            ineffective_time=90,
            priority_weight=1.0,
            service_time=6,
        )

        helicopter_state = initial_helicopter_state(self.helicopter)
        task_state = initial_task_state(task)

        self.assertEqual(helicopter_state.node_id, "BASE")
        self.assertEqual(helicopter_state.remaining_resources, self.helicopter.capacity)
        self.assertEqual(task_state.status, TaskStatus.PENDING)

    def test_future_task_starts_unrevealed(self) -> None:
        task = Task(
            task_id="T1",
            node_id="A",
            requirements=ResourceVector(personnel=1),
            optimal_time=10,
            effective_time=30,
            ineffective_time=90,
            priority_weight=1.5,
            service_time=6,
            release_time=12,
        )

        self.assertEqual(initial_task_state(task).status, TaskStatus.UNREVEALED)


if __name__ == "__main__":
    unittest.main()
