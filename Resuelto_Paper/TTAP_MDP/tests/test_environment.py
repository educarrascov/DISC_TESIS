"""Optional Gymnasium adapter smoke tests."""

import unittest

from ttap_mdp import build_small_scenario


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


if __name__ == "__main__":
    unittest.main()
