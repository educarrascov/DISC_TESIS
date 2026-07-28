"""Evaluate a trained MaskablePPO or DQN model on common episode seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..evaluation import EpisodeResult, aggregate_results, format_results_table
from .common import make_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained TTAP RL model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--algorithm", choices=("ppo", "dqn"), required=True)
    parser.add_argument("--scenario", choices=("small", "talcahuano"), default="small")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    try:
        if args.algorithm == "ppo":
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(args.model)
            policy_name = "MaskablePPO"
        else:
            from stable_baselines3 import DQN

            model = DQN.load(args.model)
            policy_name = "DQN"
    except ImportError as exc:
        raise SystemExit(
            "RL dependencies are missing. Run: pip install -e '.[rl]'"
        ) from exc

    environment = make_environment(args.scenario, args.stochastic)
    results: list[EpisodeResult] = []
    for seed in range(args.episodes):
        observation, _ = environment.reset(seed=seed)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            if args.algorithm == "ppo":
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                    action_masks=environment.action_masks(),
                )
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = environment.step(action)
            steps += 1
        summary = environment.simulator.summary()
        results.append(
            EpisodeResult(
                policy=policy_name,
                seed=seed,
                benefit=summary["benefit"],
                completed_tasks=summary["completed_tasks"],
                completion_rate=summary["completion_rate"],
                average_response_time=summary["average_response_time"],
                makespan=summary["makespan"],
                flight_time=summary["flight_time"],
                failures=summary["failures"],
                invalid_actions=summary["invalid_actions"],
                steps=steps,
            )
        )

    aggregated = aggregate_results(results)
    print(format_results_table(aggregated))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0].as_dict()))
            writer.writeheader()
            writer.writerows(result.as_dict() for result in results)
        print(f"\nSaved: {args.csv}")


if __name__ == "__main__":
    main()
