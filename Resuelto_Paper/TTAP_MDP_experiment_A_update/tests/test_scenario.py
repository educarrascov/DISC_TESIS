"""Scenario construction and validation tests."""

import unittest

from ttap_mdp import (
    CoordinateSystem,
    Helicopter,
    Node,
    ResourceVector,
    Scenario,
    ScenarioValidationError,
    Task,
    EXPERIMENT_A_TASK_COUNTS,
    build_experiment_a_scenario,
    build_small_scenario,
    build_talcahuano_scenario,
)


def one_task() -> Task:
    return Task(
        "T1",
        "A",
        ResourceVector(cargo=1),
        5,
        10,
        20,
        1,
        2,
    )


def one_helicopter() -> Helicopter:
    return Helicopter(
        "H1",
        "test",
        "BASE",
        ResourceVector(cargo=2),
        100,
        True,
    )


class ScenarioTests(unittest.TestCase):
    def test_small_scenario_structure(self) -> None:
        scenario = build_small_scenario()
        self.assertEqual(len(scenario.nodes), 4)
        self.assertEqual(len(scenario.helicopters), 2)
        self.assertEqual(len(scenario.tasks), 6)
        self.assertEqual(scenario.base.node_id, "BASE")

    def test_talcahuano_structure_matches_paper(self) -> None:
        scenario = build_talcahuano_scenario()
        self.assertEqual(scenario.scenario_id, "GPS_7areas_6helos_30tasks")
        self.assertEqual(len(scenario.nodes) - 1, 7)
        self.assertEqual(len(scenario.helicopters), 6)
        self.assertEqual(len(scenario.tasks), 30)
        self.assertEqual(scenario.horizon, 200)
        self.assertEqual(scenario.recovery_time, 2)
        self.assertTrue(
            all(
                helicopter.model == "AS532_Cougar"
                for helicopter in scenario.helicopters[-2:]
            )
        )

    def test_experiment_a_geometry_and_fleet_match_original(self) -> None:
        scenario = build_experiment_a_scenario(20, 1)
        self.assertEqual(len(scenario.nodes) - 1, 4)
        self.assertEqual(len(scenario.helicopters), 4)
        self.assertEqual(len(scenario.tasks), 20)
        self.assertEqual(
            {
                node.node_id: node.coordinates
                for node in scenario.nodes
            },
            {
                "BASE": (0.0, 0.0),
                "A": (10.0, 5.0),
                "B": (20.0, 15.0),
                "C": (30.0, 10.0),
                "D": (40.0, 20.0),
            },
        )
        self.assertEqual(
            [helicopter.speed_kmh for helicopter in scenario.helicopters],
            [150, 120, 140, 160],
        )
        self.assertEqual(
            [
                helicopter.visual_capable
                for helicopter in scenario.helicopters
            ],
            [True, False, True, False],
        )
        self.assertTrue(
            all(
                helicopter.capacity == ResourceVector(500, 3, 3)
                for helicopter in scenario.helicopters
            )
        )

    def test_experiment_a_task_sets_are_nested(self) -> None:
        full = build_experiment_a_scenario(60, 7)
        for n_tasks in EXPERIMENT_A_TASK_COUNTS:
            with self.subTest(n_tasks=n_tasks):
                prefix = build_experiment_a_scenario(n_tasks, 7)
                self.assertEqual(prefix.tasks, full.tasks[:n_tasks])
                self.assertEqual(
                    {task.node_id for task in prefix.tasks},
                    {"A", "B", "C", "D"},
                )

    def test_haversine_distance_is_symmetric(self) -> None:
        scenario = build_talcahuano_scenario()
        first = scenario.distance_km("BASE", "A")
        second = scenario.distance_km("A", "BASE")
        self.assertAlmostEqual(first, second)
        self.assertGreater(first, 0)

    def test_duplicate_identifiers_are_rejected(self) -> None:
        with self.assertRaises(ScenarioValidationError):
            Scenario(
                "duplicate",
                (
                    Node("BASE", 0, 0, is_base=True),
                    Node("A", 1, 0),
                    Node("A", 2, 0),
                ),
                (one_helicopter(),),
                (one_task(),),
            )

    def test_exactly_one_base_is_required(self) -> None:
        with self.assertRaises(ScenarioValidationError):
            Scenario(
                "no_base",
                (Node("BASE", 0, 0), Node("A", 1, 0)),
                (one_helicopter(),),
                (one_task(),),
            )

    def test_mixed_coordinate_systems_are_rejected(self) -> None:
        with self.assertRaises(ScenarioValidationError):
            Scenario(
                "mixed",
                (
                    Node("BASE", 0, 0, is_base=True),
                    Node(
                        "A",
                        -36,
                        -73,
                        coordinate_system=CoordinateSystem.GEOGRAPHIC_DEGREES,
                    ),
                ),
                (one_helicopter(),),
                (one_task(),),
            )

    def test_unknown_task_node_is_rejected(self) -> None:
        bad_task = Task(
            "T1",
            "UNKNOWN",
            ResourceVector(cargo=1),
            5,
            10,
            20,
            1,
            2,
        )
        with self.assertRaises(ScenarioValidationError):
            Scenario(
                "bad_reference",
                (Node("BASE", 0, 0, is_base=True), Node("A", 1, 0)),
                (one_helicopter(),),
                (bad_task,),
            )


if __name__ == "__main__":
    unittest.main()
