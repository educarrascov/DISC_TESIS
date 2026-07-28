"""Event simulator and uncertainty tests."""

import unittest

from ttap_mdp import (
    ActionKind,
    HelicopterStatus,
    Simulator,
    TaskStatus,
    UncertaintyConfig,
    build_small_scenario,
)


class DynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = build_small_scenario()

    def test_action_catalog_has_fixed_expected_size(self) -> None:
        simulator = Simulator(self.scenario)
        expected = 1 + len(self.scenario.helicopters) + (
            len(self.scenario.helicopters) * len(self.scenario.tasks)
        )
        self.assertEqual(len(simulator.actions), expected)
        self.assertTrue(simulator.action_mask()[0])

    def test_assignment_completes_and_consumes_resources(self) -> None:
        simulator = Simulator(self.scenario, strict_actions=True)
        index = simulator.action_index(ActionKind.ASSIGN, "H1", "T1")
        simulator.step(index)
        while simulator.task_states["T1"].status is not TaskStatus.COMPLETED:
            simulator.step(0)
        state = simulator.helicopter_states["H1"]
        self.assertEqual(state.status, HelicopterStatus.AVAILABLE)
        self.assertEqual(state.node_id, "A")
        self.assertEqual(state.remaining_resources.cargo, 200)
        self.assertGreater(simulator.cumulative_reward, 0)

    def test_return_recovers_full_capacity(self) -> None:
        simulator = Simulator(self.scenario, strict_actions=True)
        simulator.step(simulator.action_index(ActionKind.ASSIGN, "H1", "T1"))
        while simulator.task_states["T1"].status is not TaskStatus.COMPLETED:
            simulator.step(0)
        simulator.step(simulator.action_index(ActionKind.RETURN, "H1"))
        while simulator.helicopter_states["H1"].status is not HelicopterStatus.AVAILABLE:
            simulator.step(0)
        state = simulator.helicopter_states["H1"]
        self.assertEqual(state.node_id, "BASE")
        self.assertEqual(
            state.remaining_resources,
            self.scenario.helicopter_by_id["H1"].capacity,
        )

    def test_certain_failure_preserves_pending_task_and_recovers(self) -> None:
        uncertainty = UncertaintyConfig(
            failure_probability=1.0,
            downtime_min=5,
            downtime_max=5,
        )
        simulator = Simulator(self.scenario, uncertainty, seed=10)
        simulator.step(simulator.action_index(ActionKind.ASSIGN, "H1", "T1"))
        self.assertEqual(simulator.failures, 1)
        self.assertEqual(simulator.task_states["T1"].status, TaskStatus.PENDING)
        self.assertEqual(
            simulator.helicopter_states["H1"].status,
            HelicopterStatus.UNAVAILABLE,
        )
        simulator.step(0)
        self.assertEqual(
            simulator.helicopter_states["H1"].status,
            HelicopterStatus.AVAILABLE,
        )

    def test_dynamic_arrivals_are_reproducible(self) -> None:
        uncertainty = UncertaintyConfig(
            dynamic_arrival_window=40,
            initial_task_fraction=0,
        )
        first = Simulator(self.scenario, uncertainty, seed=218)
        second = Simulator(self.scenario, uncertainty, seed=218)
        self.assertEqual(first.release_times, second.release_times)
        self.assertTrue(
            all(
                state.status is TaskStatus.UNREVEALED
                for state in first.task_states.values()
            )
        )

    def test_stochastic_episode_reset_is_seeded(self) -> None:
        uncertainty = UncertaintyConfig.moderate()
        first = Simulator(self.scenario, uncertainty, seed=5)
        second = Simulator(self.scenario, uncertainty, seed=5)
        self.assertEqual(first.release_times, second.release_times)
        first_action = next(
            i for i, allowed in enumerate(first.action_mask()[1:], start=1) if allowed
        )
        second_action = next(
            i for i, allowed in enumerate(second.action_mask()[1:], start=1) if allowed
        )
        first.step(first_action)
        second.step(second_action)
        self.assertEqual(first.log, second.log)

    def test_invalid_unmasked_action_is_safe(self) -> None:
        simulator = Simulator(self.scenario, strict_actions=False)
        invalid_return = simulator.action_index(ActionKind.RETURN, "H1")
        result = simulator.step(invalid_return)
        self.assertFalse(result.valid_action)
        self.assertEqual(simulator.invalid_actions, 1)


if __name__ == "__main__":
    unittest.main()
