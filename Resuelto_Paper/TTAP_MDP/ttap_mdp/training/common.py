"""Shared RL training helpers."""

from __future__ import annotations

from pathlib import Path

from ..dynamics import UncertaintyConfig
from ..environment import TTAPGymEnv
from ..scenario import build_small_scenario, build_talcahuano_scenario


def make_environment(
    scenario_name: str,
    stochastic: bool,
    *,
    invalid_action_penalty: float = 0.0,
) -> TTAPGymEnv:
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
    return TTAPGymEnv(
        scenario,
        uncertainty,
        invalid_action_penalty=invalid_action_penalty,
    )


def ensure_output_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
