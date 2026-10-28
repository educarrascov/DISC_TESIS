"""Optional contribution check: repeated recovery versus one sortie per asset."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint
from scipy.sparse import coo_matrix, vstack

import ttap_warm_start as native
from objective_profiles import ObjectiveSpec, apply_objective, compute_metrics


def one_sortie_package(package, helicopter_names):
    """Add sum(return arcs of h) <= 1. A used asset must still return to terminate."""
    t0 = time.perf_counter()
    names = list(helicopter_names)
    row_id = {h: i for i, h in enumerate(names)}
    selected = [(row_id[a["h"]], j) for j, a in enumerate(package["all_arcs"]) if a["kind"] == "return"]
    rr, cc = zip(*selected) if selected else ([], [])
    extra = coo_matrix((np.ones(len(selected)), (rr, cc)), shape=(len(names), len(package["c"])))
    matrix = vstack([package["A"], extra], format="csr")
    constraints = LinearConstraint(matrix, np.r_[package["constraints"].lb, np.full(len(names), -np.inf)],
                                   np.r_[package["constraints"].ub, np.ones(len(names))])
    result = dict(package)
    result.update({"A": matrix, "constraints": constraints,
                   "matrix_time": package["matrix_time"] + time.perf_counter() - t0})
    return result


def compare_recovery(output, experiment="A", tasks=20, seed=1, spec=None, solver_budget=600.):
    spec = spec or ObjectiveSpec()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new ablation output directory")
    output.mkdir(parents=True, exist_ok=True)
    kernel = native.load_kernel(experiment)
    scenario = apply_objective(kernel["create_scenario"](tasks, seed), spec)
    full = kernel["prepare_exact_problem"](scenario)
    restricted = one_sortie_package(full, scenario["helicopters"])
    rows = []
    for label, package in (("RepeatedRecovery", full), ("OneSortie", restricted)):
        t0 = time.perf_counter()
        lp = native.to_highs_lp(package)
        lp_seconds = time.perf_counter() - t0
        result, summary = native.solve_one(scenario, package, lp, solver_budget, label, lp_time=lp_seconds)
        kernel["validate_exact_incumbent"](result, summary, package)
        routes = kernel["decode_exact_solution"](scenario, result, package["all_arcs"], summary["Method"])
        row = compute_metrics(kernel, scenario, routes, summary, spec)
        row.update({"Ablation": label, "Configuration": experiment,
                    "Model SHA256": native.model_fingerprint(package)})
        rows.append(row)
        routes.to_csv(output / f"routes_{label}.csv", index=False)
    data = pd.DataFrame(rows)
    data.to_csv(output / "recovery_ablation.csv", index=False)
    lower = data["Normalized Mission Value"].fillna(0).to_numpy()
    upper = data["Dual Bound"].fillna(1).clip(upper=1).to_numpy()
    result = pd.DataFrame([{
        "Both certified": bool(data["Optimal Certified"].all()),
        "Certified recovery value gain": lower[0]-lower[1] if data["Optimal Certified"].all() else np.nan,
        "Recovery value gain lower bound": max(0., lower[0]-upper[1]),
        "Recovery value gain upper bound": max(0., upper[0]-lower[1])}])
    result.to_csv(output / "recovery_value_interval.csv", index=False)
    return data, result
