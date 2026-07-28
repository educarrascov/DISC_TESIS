"""Train the secondary DQN comparison policy."""

from __future__ import annotations

import argparse

from .common import ensure_output_directory, make_environment


def main() -> None:
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "RL dependencies are missing. Run: pip install -e '.[rl]'"
        ) from exc

    parser = argparse.ArgumentParser(description="Train DQN on TTAP-MDP.")
    parser.add_argument("--scenario", choices=("small", "talcahuano"), default="small")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=218)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()

    # Standard SB3 DQN has no action-mask support. Infeasible actions therefore
    # behave as WAIT. The default reward remains the original TTAP reward.
    environment = Monitor(
        make_environment(
            args.scenario,
            args.stochastic,
            invalid_action_penalty=0.0,
        )
    )
    model = DQN(
        "MlpPolicy",
        environment,
        seed=args.seed,
        verbose=1,
        learning_starts=2_000,
        buffer_size=50_000,
        tensorboard_log=str(ensure_output_directory(args.output) / "tensorboard"),
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    suffix = "stochastic" if args.stochastic else "deterministic"
    destination = ensure_output_directory(args.output) / (
        f"dqn_{args.scenario}_{suffix}"
    )
    model.save(destination)
    print(f"Model saved at: {destination}.zip")


if __name__ == "__main__":
    main()
