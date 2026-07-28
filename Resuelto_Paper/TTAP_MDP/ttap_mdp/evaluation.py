"""Policy execution, repeated evaluation, and compact result tables."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable, Protocol

from .dynamics import Simulator, UncertaintyConfig
from .scenario import Scenario


class Policy(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def select_action(self, simulator: Simulator) -> int: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    policy: str
    seed: int
    benefit: float
    completed_tasks: int
    completion_rate: float
    average_response_time: float
    makespan: float
    flight_time: float
    failures: int
    invalid_actions: int
    steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "benefit": self.benefit,
            "completed_tasks": self.completed_tasks,
            "completion_rate": self.completion_rate,
            "average_response_time": self.average_response_time,
            "makespan": self.makespan,
            "flight_time": self.flight_time,
            "failures": self.failures,
            "invalid_actions": self.invalid_actions,
            "steps": self.steps,
        }


def run_episode(
    scenario: Scenario,
    policy: Policy,
    *,
    uncertainty: UncertaintyConfig | None = None,
    seed: int = 0,
    max_steps: int = 20_000,
) -> tuple[EpisodeResult, Simulator]:
    simulator = Simulator(
        scenario,
        uncertainty,
        seed=seed,
        strict_actions=True,
    )
    policy.reset(seed)
    steps = 0
    while not simulator.terminated:
        if steps >= max_steps:
            raise RuntimeError("Episode exceeded max_steps.")
        simulator.step(policy.select_action(simulator))
        steps += 1
    summary = simulator.summary()
    result = EpisodeResult(
        policy=policy.name,
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
    return result, simulator


def evaluate_policy(
    scenario: Scenario,
    policy_factory: Callable[[], Policy],
    *,
    uncertainty: UncertaintyConfig | None = None,
    seeds: range | list[int] | tuple[int, ...] = range(10),
) -> list[EpisodeResult]:
    return [
        run_episode(
            scenario,
            policy_factory(),
            uncertainty=uncertainty,
            seed=seed,
        )[0]
        for seed in seeds
    ]


def aggregate_results(results: list[EpisodeResult]) -> list[dict[str, object]]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.policy, []).append(result)

    rows: list[dict[str, object]] = []
    for policy_name, policy_results in grouped.items():
        benefits = [result.benefit for result in policy_results]
        completed = [result.completed_tasks for result in policy_results]
        response = [result.average_response_time for result in policy_results]
        rows.append(
            {
                "policy": policy_name,
                "episodes": len(policy_results),
                "benefit_mean": mean(benefits),
                "benefit_sd": stdev(benefits) if len(benefits) > 1 else 0.0,
                "completed_mean": mean(completed),
                "response_time_mean": mean(response),
                "failures_mean": mean(result.failures for result in policy_results),
            }
        )
    return sorted(rows, key=lambda row: -float(row["benefit_mean"]))


def format_results_table(rows: list[dict[str, object]]) -> str:
    headers = (
        "Policy",
        "N",
        "Benefit mean",
        "Benefit SD",
        "Completed",
        "Response",
        "Failures",
    )
    rendered = [
        (
            str(row["policy"]),
            str(row["episodes"]),
            f'{float(row["benefit_mean"]):.4f}',
            f'{float(row["benefit_sd"]):.4f}',
            f'{float(row["completed_mean"]):.2f}',
            f'{float(row["response_time_mean"]):.2f}',
            f'{float(row["failures_mean"]):.2f}',
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headers))
    ]
    line = " | ".join(
        headers[index].ljust(widths[index]) for index in range(len(headers))
    )
    separator = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in rendered
    ]
    return "\n".join([line, separator, *body])
