"""Train the primary action-masked PPO policy."""

from __future__ import annotations

import argparse

from .common import ensure_output_directory, make_environment


def main() -> None:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "RL dependencies are missing. Run: pip install -e '.[rl]'"
        ) from exc

    parser = argparse.ArgumentParser(description="Train MaskablePPO on TTAP-MDP.")
    parser.add_argument("--scenario", choices=("small", "talcahuano"), default="small")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=218)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()

    base_environment = make_environment(args.scenario, args.stochastic)
    environment = Monitor(
        ActionMasker(
            base_environment,
            lambda env: env.action_masks(),
        )
    )
    model = MaskablePPO(
        "MlpPolicy",
        environment,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(ensure_output_directory(args.output) / "tensorboard"),
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    suffix = "stochastic" if args.stochastic else "deterministic"
    destination = ensure_output_directory(args.output) / (
        f"maskable_ppo_{args.scenario}_{suffix}"
    )
    model.save(destination)
    print(f"Model saved at: {destination}.zip")


if __name__ == "__main__":
    main()
