"""Command-line baseline demonstration with no third-party dependencies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .baselines import (
    GreedyOnlinePolicy,
    RandomFeasiblePolicy,
    RollingHorizonPolicy,
)
from .dynamics import UncertaintyConfig
from .evaluation import aggregate_results, evaluate_policy, format_results_table
from .scenario import build_small_scenario, build_talcahuano_scenario


def run_demo(
    *,
    scenario_name: str = "talcahuano",
    episodes: int = 10,
    stochastic: bool = False,
) -> list[dict[str, object]]:
    scenario = (
        build_talcahuano_scenario()
        if scenario_name == "talcahuano"
        else build_small_scenario()
    )
    uncertainty = (
        UncertaintyConfig.moderate()
        if stochastic
        else UncertaintyConfig.deterministic()
    )
    factories = (
        lambda: RandomFeasiblePolicy(),
        GreedyOnlinePolicy,
        RollingHorizonPolicy,
    )
    results = []
    for factory in factories:
        results.extend(
            evaluate_policy(
                scenario,
                factory,
                uncertainty=uncertainty,
                seeds=range(episodes),
            )
        )
    return aggregate_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TTAP-MDP v1.0 baselines.")
    parser.add_argument(
        "--scenario",
        choices=("small", "talcahuano"),
        default="talcahuano",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = run_demo(
        scenario_name=args.scenario,
        episodes=args.episodes,
        stochastic=args.stochastic,
    )
    print(format_results_table(rows))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {args.csv}")


if __name__ == "__main__":
    main()
