"""Optional Gymnasium adapter smoke tests."""

import unittest

from ttap_mdp import (
    ExperimentAFamilyEnv,
    UncertaintyConfig,
    build_experiment_a_scenario,
    build_small_scenario,
)


try:
    import gymnasium  # noqa: F401
except ImportError:
    gymnasium = None


@unittest.skipIf(gymnasium is None, "Gymnasium optional dependency is not installed")
class EnvironmentTests(unittest.TestCase):
    def test_environment_shapes_and_step_signature(self) -> None:
        from ttap_mdp.environment import TTAPGymEnv

        environment = TTAPGymEnv(build_small_scenario())
        observation, info = environment.reset(seed=218)
        self.assertTrue(environment.observation_space.contains(observation))
        self.assertEqual(len(info["action_mask"]), environment.action_space.n)
        result = environment.step(0)
        self.assertEqual(len(result), 5)

    def test_experiment_a_family_changes_instance_not_spaces(self) -> None:
        environment = ExperimentAFamilyEnv(
            20,
            (1001, 1002, 1003),
            UncertaintyConfig.deterministic(),
            sampler_seed=218,
        )
        first_observation, first_info = environment.reset(seed=218)
        first_action_size = environment.action_space.n
        second_observation, second_info = environment.reset()
        self.assertEqual(first_observation.shape, second_observation.shape)
        self.assertEqual(environment.action_space.n, first_action_size)
        self.assertIn(first_info["instance_seed"], (1001, 1002, 1003))
        self.assertIn(second_info["instance_seed"], (1001, 1002, 1003))

    def test_environment_can_swap_compatible_experiment_a_instances(self) -> None:
        from ttap_mdp.environment import TTAPGymEnv

        environment = TTAPGymEnv(build_experiment_a_scenario(30, 1))
        original_shape = environment.observation_space.shape
        original_actions = environment.action_space.n
        environment.set_scenario(build_experiment_a_scenario(30, 2))
        observation, _ = environment.reset(seed=1)
        self.assertEqual(observation.shape, original_shape)
        self.assertEqual(environment.action_space.n, original_actions)


if __name__ == "__main__":
    unittest.main()
