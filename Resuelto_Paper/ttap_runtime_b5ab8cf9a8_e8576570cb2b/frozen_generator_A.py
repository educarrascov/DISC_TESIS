"""Frozen generator A; source commit 07dd90042822e2cab84209fe81c9f31ecfe230a8.
Extracted from the recovered notebook without executing its analysis cells.
Only the primary-output docstring is clarified; numerical definitions are unchanged.
"""

import math

import os

import platform

import sys

import time

from collections import Counter, defaultdict, deque

from datetime import datetime

from zoneinfo import ZoneInfo

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy

from scipy.optimize import Bounds, LinearConstraint, milp

from scipy.sparse import coo_matrix

TASK_COUNTS = [20, 30, 40, 50, 60]

INSTANCE_SEEDS = list(range(1, 31))

MAX_TASKS = max(TASK_COUNTS)

TASK_GENERATOR_VERSION = "matched_v3_impact_anytime"

TASK_PROFILE_STREAM_ID = 21001

TASK_LOCATION_STREAM_A_ID = 22001

TASK_CLASS_NAMES = np.array(["medical", "personnel", "cargo"])

TASK_CLASS_PROBABILITIES = np.array([0.45, 0.30, 0.25])

VISUAL_REQUIRED_PROBABILITY = 0.40

DEFAULT_PARAMS = {
    "SERVICE_TIME": 6.0,
    "RECOVERY_TIME": 2.0,
    "TMAX": 200.0,
    "ALPHA": 0.1,
    "TIME_UNIT": 1.0,
    "USE_PRIORITY_WEIGHTS": True,
    "PRIORITY_WEIGHTS": {
        "medical": 2.0,
        "personnel": 1.5,
        "cargo": 1.0,
    },
    # Fixed class-specific references used both to make q_i dimensionless
    # and to establish the relative scale between task classes.
    "IMPACT_REFERENCES": {
        "medical": 3.0,      # medical evacuees
        "personnel": 3.0,    # transported people
        "cargo": 500.0,      # kg delivered
    },
}

def default_nodes():
    return {
        "Base": (0.0, 0.0),
        "A": (10.0, 5.0),
        "B": (20.0, 15.0),
        "C": (30.0, 10.0),
        "D": (40.0, 20.0),
    }

def default_fleet():
    return {
        "H1": {"visual": True,  "load": 500, "med": 3, "pers": 3, "speed": 150},
        "H2": {"visual": False, "load": 500, "med": 3, "pers": 3, "speed": 120},
        "H3": {"visual": True,  "load": 500, "med": 3, "pers": 3, "speed": 140},
        "H4": {"visual": False, "load": 500, "med": 3, "pers": 3, "speed": 160},
    }

def classify_task_priority(task):
    if task["med"] > 0:
        return "medical"
    if task["pers"] > 0:
        return "personnel"
    if task["load"] > 0:
        return "cargo"
    raise ValueError("Task has no positive operational requirement.")

def raw_humanitarian_impact(task, task_class):
    """Primary humanitarian output e_i; auxiliary operational requirements are separate."""
    if task_class == "medical":
        return float(task["med"]), "medical evacuees"
    if task_class == "personnel":
        return float(task["pers"]), "people transported"
    if task_class == "cargo":
        return float(task["load"]), "kg delivered"
    raise ValueError(f"Unknown task class: {task_class}")

def assign_task_value_metadata(scenario):
    params = scenario["params"]
    weights = params["PRIORITY_WEIGHTS"]
    references = params["IMPACT_REFERENCES"]
    use_weights = params["USE_PRIORITY_WEIGHTS"]

    for task in scenario["tasks"].values():
        task_class = classify_task_priority(task)
        raw_impact, impact_unit = raw_humanitarian_impact(task, task_class)
        reference = float(references[task_class])
        if reference <= 0:
            raise ValueError("Every impact reference must be positive.")

        task["priority_class"] = task_class
        task["class_weight"] = float(weights[task_class]) if use_weights else 1.0
        task["raw_impact"] = raw_impact
        task["impact_unit"] = impact_unit
        task["impact_reference"] = reference
        task["impact"] = raw_impact / reference
        task["potential_value"] = task["class_weight"] * task["impact"]

    total_potential_value = sum(
        task["potential_value"] for task in scenario["tasks"].values()
    )
    if total_potential_value <= 0:
        raise ValueError("Total potential mission value must be positive.")

    scenario["total_potential_value"] = float(total_potential_value)
    return scenario

def make_reproducible_rng(seed, stream_id):
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(stream_id)])
    )

REFERENCE_FLEET = {
    h: {k: v for k, v in data.items() if k != "speed"}
    for h, data in default_fleet().items()
}

def task_is_structurally_feasible(task, fleet=REFERENCE_FLEET):
    return any(
        (not task["visual"] or helo["visual"])
        and helo["load"] >= task["load"]
        and helo["med"] >= task["med"]
        and helo["pers"] >= task["pers"]
        for helo in fleet.values()
    )

def draw_task_profile(rng):
    task_class = str(rng.choice(TASK_CLASS_NAMES, p=TASK_CLASS_PROBABILITIES))
    visual = bool(rng.random() < VISUAL_REQUIRED_PROBABILITY)

    if task_class == "medical":
        load = int(rng.choice(np.arange(50, 501, 50)))
        med = int(rng.integers(1, 4))
        pers = int(rng.integers(0, 4))
    elif task_class == "personnel":
        load = int(rng.choice(np.arange(50, 501, 50)))
        med = 0
        pers = int(rng.integers(1, 4))
    else:
        load = int(rng.choice(np.arange(100, 501, 50)))
        med = 0
        pers = 0

    t_opt = int(rng.integers(8, 36))
    t_eff = t_opt + int(rng.integers(12, 27))
    t_ineff = t_eff + int(rng.integers(20, 61))

    task = {
        "visual": visual,
        "load": load,
        "med": med,
        "pers": pers,
        "t_opt": float(t_opt),
        "t_eff": float(t_eff),
        "t_ineff": float(t_ineff),
    }
    if classify_task_priority(task) != task_class:
        raise AssertionError("Generated class is inconsistent.")
    if not task_is_structurally_feasible(task):
        raise AssertionError("Generated task is structurally infeasible.")
    return task

def task_prefix_is_complex(tasks, n_tasks):
    prefix = [tasks[i] for i in range(1, n_tasks + 1)]
    class_counts = Counter(classify_task_priority(task) for task in prefix)
    minimum_class_count = max(3, int(math.floor(0.12 * n_tasks)))
    if any(class_counts[name] < minimum_class_count for name in TASK_CLASS_NAMES):
        return False

    visual_count = sum(task["visual"] for task in prefix)
    if not (math.ceil(0.25 * n_tasks) <= visual_count <= math.floor(0.60 * n_tasks)):
        return False

    resource_vectors = {
        (task["load"], task["med"], task["pers"]) for task in prefix
    }
    return len(resource_vectors) >= math.ceil(0.65 * n_tasks)

def generate_task_profiles(n_tasks=20, seed=1):
    if n_tasks not in TASK_COUNTS:
        raise ValueError(f"n_tasks must belong to {TASK_COUNTS}.")
    rng = make_reproducible_rng(seed, TASK_PROFILE_STREAM_ID)
    for _ in range(10000):
        candidate = {
            task_id: draw_task_profile(rng)
            for task_id in range(1, MAX_TASKS + 1)
        }
        if all(task_prefix_is_complex(candidate, n) for n in TASK_COUNTS):
            return {
                task_id: candidate[task_id].copy()
                for task_id in range(1, n_tasks + 1)
            }
    raise RuntimeError("Could not generate a sufficiently heterogeneous task set.")

def generate_asymmetric_locations(n_tasks, seed, node_choices):
    node_choices = list(node_choices)
    if len(node_choices) != 4:
        raise ValueError("Experiment A requires exactly four demand areas.")
    base_probabilities = np.array([0.45, 0.30, 0.17, 0.08])
    rng = make_reproducible_rng(seed, TASK_LOCATION_STREAM_A_ID)

    for _ in range(10000):
        hotspot_order = list(rng.permutation(node_choices))
        sequence = list(
            rng.choice(
                hotspot_order,
                size=MAX_TASKS,
                replace=True,
                p=base_probabilities,
            )
        )
        valid = True
        for prefix_size in TASK_COUNTS:
            counts = Counter(sequence[:prefix_size])
            values = np.array([counts[node] for node in node_choices])
            required_imbalance = max(4, int(math.ceil(0.20 * prefix_size)))
            if not np.all(values >= 1) or values.max() - values.min() < required_imbalance:
                valid = False
                break
        if valid:
            return {
                task_id: str(sequence[task_id - 1])
                for task_id in range(1, n_tasks + 1)
            }
    raise RuntimeError("Could not generate an asymmetric spatial allocation.")

def create_scenario(n_tasks=20, seed=1, scenario_id=None, params=None):
    if scenario_id is None:
        scenario_id = f"A4x4_T{n_tasks}_S{seed}"
    merged_params = DEFAULT_PARAMS.copy()
    merged_params["PRIORITY_WEIGHTS"] = DEFAULT_PARAMS["PRIORITY_WEIGHTS"].copy()
    merged_params["IMPACT_REFERENCES"] = DEFAULT_PARAMS["IMPACT_REFERENCES"].copy()
    if params:
        merged_params.update(params)

    nodes = default_nodes()
    tasks = generate_task_profiles(n_tasks=n_tasks, seed=seed)
    task_to_node = generate_asymmetric_locations(
        n_tasks=n_tasks,
        seed=seed,
        node_choices=[node for node in nodes if node != "Base"],
    )
    scenario = {
        "scenario_id": scenario_id,
        "experiment": "A",
        "task_generator_version": TASK_GENERATOR_VERSION,
        "seed": seed,
        "nodes": nodes,
        "helicopters": default_fleet(),
        "tasks": tasks,
        "task_to_node": task_to_node,
        "params": merged_params,
    }
    return assign_task_value_metadata(scenario)

def validate_scenario_structure(scenario):
    if len(scenario["helicopters"]) != 4:
        raise AssertionError("Experiment A must contain four helicopters.")
    if len([n for n in scenario["nodes"] if n != "Base"]) != 4:
        raise AssertionError("Experiment A must contain four demand areas.")
    for task_id, task in scenario["tasks"].items():
        if not task_is_structurally_feasible(task, scenario["helicopters"]):
            raise AssertionError(f"Task {task_id} is structurally infeasible.")
        if not (0 < task["impact"] <= 1):
            raise AssertionError(f"Task {task_id} has invalid normalized impact.")
        if not math.isclose(
            task["potential_value"],
            task["class_weight"] * task["impact"],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Task {task_id} has inconsistent value metadata.")
    return True
