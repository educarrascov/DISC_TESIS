"""Validate the optional Gymnasium interface against Gymnasium and SB3."""

from __future__ import annotations

from .common import make_environment


def main() -> None:
    try:
        from gymnasium.utils.env_checker import check_env as gym_check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check_env
    except ImportError as exc:
        raise SystemExit(
            "RL dependencies are missing. Run: pip install -e '.[rl]'"
        ) from exc

    environment = make_environment("small", stochastic=True)
    gym_check_env(environment, skip_render_check=True)
    sb3_check_env(environment, warn=True)
    observation, info = environment.reset(seed=218)
    assert environment.observation_space.contains(observation)
    assert len(info["action_mask"]) == environment.action_space.n
    print("Gymnasium and Stable-Baselines3 checks passed.")


if __name__ == "__main__":
    main()
