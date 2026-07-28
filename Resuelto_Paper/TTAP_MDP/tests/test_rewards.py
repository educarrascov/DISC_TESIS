"""Temporal reward tests."""

import unittest

from ttap_mdp import (
    ResourceVector,
    Task,
    build_small_scenario,
    normalized_task_benefit,
    temporal_satisfaction,
)


class RewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = Task(
            "T",
            "A",
            ResourceVector(cargo=1),
            optimal_time=10,
            effective_time=30,
            ineffective_time=60,
            priority_weight=2,
            service_time=6,
        )

    def test_satisfaction_anchor_values(self) -> None:
        self.assertEqual(temporal_satisfaction(0, self.task), 1.0)
        self.assertEqual(temporal_satisfaction(10, self.task), 1.0)
        self.assertAlmostEqual(temporal_satisfaction(30, self.task), 0.5)
        self.assertEqual(temporal_satisfaction(60, self.task), 0.0)

    def test_satisfaction_is_monotone(self) -> None:
        values = [temporal_satisfaction(time, self.task) for time in range(0, 65)]
        self.assertTrue(
            all(first >= second for first, second in zip(values, values[1:]))
        )

    def test_normalized_benefits_cannot_exceed_one_in_total(self) -> None:
        scenario = build_small_scenario()
        maximum = sum(
            normalized_task_benefit(scenario, task, 0)
            for task in scenario.tasks
        )
        self.assertAlmostEqual(maximum, 1.0)


if __name__ == "__main__":
    unittest.main()
