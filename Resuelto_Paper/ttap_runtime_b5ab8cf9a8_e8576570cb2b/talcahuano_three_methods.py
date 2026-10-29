"""Reevaluate the fixed GPS case with BOTH greedy policies and native HiGHS.

Original cell-6 data are frozen separately. This helper does not require the
mapping/geopandas cells and does not change the seven GPS coordinates.
"""

from copy import deepcopy
import ast
import json
import math
from pathlib import Path
import time

import pandas as pd

import ttap_warm_start as native
from objective_profiles import ObjectiveSpec, apply_objective, compute_metrics
from public_sources import gps_notebook


ROOT = Path(__file__).resolve().parent


def original_gps_data():
    """Extract only literal data and the five reviewed pure constructors."""
    notebook = gps_notebook()
    source = notebook["cells"][6]["source"]
    tree = ast.parse("".join(source) if isinstance(source, list) else source)
    constants = {"GPS_NODES_FULL", "DEMAND_NODE_ORDER", "BASE_TASK_ROWS", "BASE_TASK_TO_NODE", "FLEET_ROWS"}
    functions = {"build_active_nodes", "build_task_rows", "build_tasks", "build_task_to_node", "build_helicopters"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
            ast.literal_eval(node.value)
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.For) and "BASE_TASK_TO_NODE" in ast.unparse(node):
            selected.append(node)
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 "verified original GPS constructors", "exec"), ns)
    names, nodes = ns["build_active_nodes"](7)
    tasks = ns["build_tasks"](ns["build_task_rows"](30))
    return {"nodes": nodes, "tasks": tasks, "task_to_node": ns["build_task_to_node"](tasks, names),
            "helicopters": ns["build_helicopters"](6)}


def gps_kernel_and_scenario(spec=None):
    spec = spec or ObjectiveSpec()
    original = original_gps_data()
    kernel = native.load_kernel("B")

    def geodesic_km(scenario, start, end):
        lat1, lon1 = map(math.radians, scenario["nodes"][start])
        lat2, lon2 = map(math.radians, scenario["nodes"][end])
        value = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
        return 6371.0 * 2 * math.atan2(math.sqrt(value), math.sqrt(max(0., 1-value)))

    kernel["euclidean_distance"] = geodesic_km
    scenario = {"scenario_id": "GPS_7areas_6helos_30tasks", "experiment": "GPS", "seed": 1,
                "task_generator_version": "GPS_fixed_cell6_07dd900",
                "nodes": original["nodes"], "helicopters": original["helicopters"],
                "tasks": {int(k): v for k, v in original["tasks"].items()},
                "task_to_node": {int(k): v for k, v in original["task_to_node"].items()},
                "params": deepcopy(kernel["DEFAULT_PARAMS"])}
    apply_objective(scenario, spec)
    kernel["validate_temporal_satisfaction"](scenario)
    return kernel, scenario


def compare_talcahuano(output, spec=None, solver_budget=600.):
    """Three fresh methods under one objective; a reference is exact only if certified."""
    spec = spec or ObjectiveSpec()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new Talcahuano output directory")
    output.mkdir(parents=True, exist_ok=True)
    kernel, scenario = gps_kernel_and_scenario(spec)
    rows, routes = [], []
    for rule in ("benefit", "density_operational"):
        frame, summary = kernel["solve_greedy_ttap"](scenario, rule=rule)
        kernel["validate_solution_consistency"](scenario, frame, summary["Method"])
        rows.append(compute_metrics(kernel, scenario, frame, summary, spec))
        routes.append(frame)
    package = kernel["prepare_exact_problem"](scenario)
    t0 = time.perf_counter()
    lp = native.to_highs_lp(package)
    lp_seconds = time.perf_counter() - t0
    result, summary = native.solve_one(scenario, package, lp, solver_budget,
                                       "ColdReference", lp_time=lp_seconds)
    kernel["validate_exact_incumbent"](result, summary, package)
    frame = kernel["decode_exact_solution"](scenario, result, package["all_arcs"], summary["Method"])
    kernel["validate_solution_consistency"](scenario, frame, summary["Method"])
    rows.append(compute_metrics(kernel, scenario, frame, summary, spec))
    routes.append(frame)
    table = pd.DataFrame(rows)
    table["Source Commit"] = native.SOURCE_COMMIT
    table["Scenario SHA256"] = native.scenario_fingerprint(scenario)
    table.to_csv(output / "talcahuano_metrics.csv", index=False)
    pd.concat(routes, ignore_index=True).to_csv(output / "talcahuano_routes.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({"objective": spec.as_dict(),
        "source_commit": native.SOURCE_COMMIT, "source_blob": "eac2e8b11e3436a2c0e378256a8db5cb6cfc5578",
        "highs": summary["HiGHS Version"], "model_sha256": native.model_fingerprint(package),
        "solver_allowance_seconds": solver_budget}, indent=2))
    return table
