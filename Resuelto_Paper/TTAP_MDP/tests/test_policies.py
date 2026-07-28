"""Baseline policy and deterministic bridge tests."""

import unittest

from ttap_mdp import (
    UncertaintyConfig,
    build_small_scenario,
    build_talcahuano_scenario,
    run_episode,
)
from ttap_mdp.baselines import (
    GreedyOnlinePolicy,
    RandomFeasiblePolicy,
    RollingHorizonPolicy,
)


class PolicyTests(unittest.TestCase):
    def test_all_baselines_terminate_on_small_scenario(self) -> None:
        scenario = build_small_scenario()
        for policy in (
            RandomFeasiblePolicy(1),
            GreedyOnlinePolicy(),
            RollingHorizonPolicy(),
        ):
            with self.subTest(policy=policy.name):
                result, simulator = run_episode(scenario, policy, seed=1)
                self.assertTrue(simulator.terminated)
                self.assertGreater(result.completed_tasks, 0)
                self.assertEqual(result.invalid_actions, 0)

    def test_talcahuano_greedy_reproduces_paper_result(self) -> None:
        result, simulator = run_episode(
            build_talcahuano_scenario(),
            GreedyOnlinePolicy(),
            uncertainty=UncertaintyConfig.deterministic(),
            seed=218,
        )
        self.assertAlmostEqual(result.benefit, 0.5930769724427806)
        self.assertEqual(result.completed_tasks, 26)
        self.assertEqual(result.makespan, 117)
        self.assertAlmostEqual(result.flight_time, 126.6447425156069)
        incomplete = {
            task_id
            for task_id, state in simulator.task_states.items()
            if state.status.value != "completed"
        }
        self.assertEqual(incomplete, {"T3", "T11", "T23", "T29"})

    def test_stochastic_repeated_run_is_reproducible(self) -> None:
        scenario = build_small_scenario()
        uncertainty = UncertaintyConfig.moderate()
        first, _ = run_episode(
            scenario,
            GreedyOnlinePolicy(),
            uncertainty=uncertainty,
            seed=9,
        )
        second, _ = run_episode(
            scenario,
            GreedyOnlinePolicy(),
            uncertainty=uncertainty,
            seed=9,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
