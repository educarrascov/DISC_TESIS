"""Independent cold/greedy-start TTAP experiments using the native HiGHS API.

No existing notebook or result is overwritten. The frozen mathematical
definitions are from commit 07dd90042822e2cab84209fe81c9f31ecfe230a8.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import highspy
import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "07dd90042822e2cab84209fe81c9f31ecfe230a8"
OBJECTIVE_VERSION = "primary-output_matched-v3_fixed-references-3-3-500"
TOL = 1e-6


def load_kernel(experiment):
    """Load definitions only, and give every generated MILP a stable order."""
    ns = runpy.run_path(str(ROOT / f"frozen_generator_{experiment}.py"))
    path = ROOT / "frozen_ttap_core.py"
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    original_builder = ns["build_state_graph_for_helicopter"]

    def stable_builder(scenario, helicopter):
        states, arcs = original_builder(scenario, helicopter)
        states.sort(key=repr)
        arcs.sort(key=lambda a: (repr(a["from"]), a["kind"],
                                 -1 if a["task"] is None else int(a["task"]),
                                 repr(a["to"])))
        return states, arcs

    ns["build_state_graph_for_helicopter"] = stable_builder
    return ns


def scenario_fingerprint(scenario):
    payload = {k: scenario[k] for k in
               ("experiment", "task_generator_version", "seed", "nodes",
                "helicopters", "tasks", "task_to_node", "params")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def model_fingerprint(package):
    matrix = package["A"].tocsr()
    digest = hashlib.sha256()
    for data in (matrix.indptr, matrix.indices, matrix.data, package["c"],
                 package["bounds"].lb, package["bounds"].ub,
                 package["constraints"].lb, package["constraints"].ub):
        values = np.ascontiguousarray(data)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def validate_vector(package, vector, tol=TOL):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (len(package["c"]),) or not np.isfinite(vector).all():
        raise ValueError("Incorrect or non-finite start vector")
    if np.max(np.abs(vector - np.rint(vector))) > tol:
        raise ValueError("Start vector is not integral")
    lower, upper = package["bounds"].lb, package["bounds"].ub
    if np.any(vector < lower - tol) or np.any(vector > upper + tol):
        raise ValueError("Start vector violates variable bounds")
    lhs = np.asarray(package["A"] @ vector).ravel()
    lower, upper = package["constraints"].lb, package["constraints"].ub
    if np.any(lhs[np.isfinite(lower)] < lower[np.isfinite(lower)] - tol):
        raise ValueError("Start vector violates lower constraint bounds")
    if np.any(lhs[np.isfinite(upper)] > upper[np.isfinite(upper)] + tol):
        raise ValueError("Start vector violates upper constraint bounds")
    return vector


def logs_to_start_vectors(kernel, scenario, package, logs):
    """Map all complete route logs in a single scan of the arc list.

    Return/recovery and termination arcs, including idle aircraft, are included.
    Only requested transitions are indexed: this avoids a second million-arc
    dictionary on large B instances.
    """
    requested = {}
    task_totals = {}
    time_unit = scenario["params"]["TIME_UNIT"]
    for label, frame in logs.items():
        kernel["validate_solution_consistency"](scenario, frame, label)
        transitions = []
        task_totals[label] = (float(frame.loc[frame["Status"].eq("Task Executed"),
                                            "Benefit"].sum())
                              if frame is not None and not frame.empty else 0.0)
        for h in scenario["helicopters"]:
            current = kernel["start_state"](scenario, h)
            rows = [] if frame is None or frame.empty else frame[
                frame["Helo ID"].eq(kernel["helo_to_int"](h))
            ].sort_values(["Start Time", "Completion Time"], kind="stable").to_dict("records")
            for row in rows:
                if row["Status"] == "Task Executed":
                    kind, task = "task", int(row["Task ID"])
                elif row["Status"] == "Return/Recovery":
                    kind, task = "return", None
                else:
                    raise ValueError(f"Unknown route status: {row['Status']}")
                if row["Start Node"] != current[0] or not math.isclose(
                    float(row["Start Time"]), current[1] * time_unit, abs_tol=1e-8
                ):
                    raise ValueError("Route discontinuity or incompatible time grid")
                scaled_time = float(row["Completion Time"]) / time_unit
                if not math.isclose(scaled_time, round(scaled_time), abs_tol=1e-8):
                    raise ValueError("Completion time is not on the model time grid")
                for resource in ("Remaining Load", "Remaining Medical", "Remaining Personnel"):
                    if not math.isclose(float(row[resource]), round(float(row[resource])), abs_tol=1e-8):
                        raise ValueError("Noninteger remaining resource in the route log")
                next_state = kernel["make_state"](
                    row["Node"], round(scaled_time), row["Remaining Load"],
                    row["Remaining Medical"], row["Remaining Personnel"])
                transitions.append((h, current, next_state, kind, task))
                current = next_state
            transitions.append((h, current, kernel["sink_state"](), "end", None))
        if len(transitions) != len(set(transitions)):
            raise ValueError("Duplicate route transition")
        requested[label] = set(transitions)

    needed = set().union(*requested.values())
    matches = {}
    for index, arc in enumerate(package["all_arcs"]):
        key = (arc["h"], arc["from"], arc["to"], arc["kind"], arc["task"])
        if key in needed:
            if key in matches:
                raise ValueError("Ambiguous state-arc transition")
            matches[key] = index
    missing = needed - matches.keys()
    if missing:
        raise ValueError(f"{len(missing)} route transitions do not exist in this MILP")

    starts = {}
    for label, transitions in requested.items():
        vector = np.zeros(len(package["c"]), dtype=float)
        vector[[matches[key] for key in transitions]] = 1.0
        validate_vector(package, vector)
        value = -float(package["c"] @ vector)
        if not math.isclose(value, task_totals[label], abs_tol=1e-8):
            raise ValueError("Start value differs from its route log")
        starts[label] = vector
    return starts


def to_highs_lp(package):
    matrix = package["A"].tocsc()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = map(int, matrix.shape[::-1])
    lp.col_cost_ = np.asarray(package["c"], dtype=np.float64)
    lp.col_lower_ = np.asarray(package["bounds"].lb, dtype=np.float64)
    lp.col_upper_ = np.asarray(package["bounds"].ub, dtype=np.float64)
    lp.row_lower_ = np.asarray(package["constraints"].lb, dtype=np.float64)
    lp.row_upper_ = np.asarray(package["constraints"].ub, dtype=np.float64)
    lp.integrality_ = [highspy.HighsVarType.kInteger] * lp.num_col_
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
    lp.a_matrix_.value_ = matrix.data.astype(np.float64)
    return lp


def solve_one(scenario, package, lp, budget, label, start=None,
              heuristic_time=0.0, mapping_time=0.0, lp_time=0.0,
              threads=1, solver_seed=0, mip_rel_gap=1e-6, verbose=False,
              previous_solver_time=0.0, previous_budget=0.0, previous_end_to_end=0.0):
    """One fresh solver per budget; no state is carried between budgets.

    A validated supplied route remains available if a very short solve stops
    during presolve before returning an incumbent. That fallback is explicitly
    distinguished from an incumbent returned by HiGHS.
    """
    highs = highspy.Highs()
    for option, value in (("output_flag", bool(verbose)), ("time_limit", float(budget)),
                          ("mip_rel_gap", float(mip_rel_gap)), ("threads", int(threads)),
                          ("random_seed", int(solver_seed))):
        if highs.setOptionValue(option, value) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"Unsupported HiGHS option: {option}")
    t0 = time.perf_counter()
    if highs.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("HiGHS could not load the MILP")
    model_load_time = time.perf_counter() - t0
    submission_time = 0.0
    submitted = False
    start_value = np.nan
    if start is not None:
        start_value = -float(package["c"] @ start)
        solution = highspy.HighsSolution()
        solution.col_value = np.asarray(start, dtype=np.float64)
        solution.value_valid = True
        t0 = time.perf_counter()
        status = highs.setSolution(solution)
        submission_time = time.perf_counter() - t0
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS could not receive the {label} start")
        submitted = True

    t0 = time.perf_counter()
    status = highs.run()
    solver_wall_time = time.perf_counter() - t0
    if status not in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning):
        raise RuntimeError(f"HiGHS failed: {status}")
    info, solution, model_status = highs.getInfo(), highs.getSolution(), highs.getModelStatus()
    solver_has = (bool(solution.value_valid) and
                  info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible)
    solver_value = float(-info.objective_function_value) if solver_has else np.nan
    vector = np.asarray(solution.col_value, dtype=float) if solver_has else None
    if vector is not None:
        validate_vector(package, vector)
    available_source = "HiGHS" if solver_has else "None"
    if start is not None and (vector is None or start_value > solver_value + 1e-9):
        vector = start.copy()
        available_source = "Validated start fallback"
    has_available = vector is not None
    value = -float(package["c"] @ vector) if has_available else np.nan
    certified = model_status == highspy.HighsModelStatus.kOptimal
    if certified and (not solver_has or not math.isclose(value, solver_value, abs_tol=TOL)):
        raise AssertionError("Certified status inconsistent with the available solution")
    raw_dual = float(info.mip_dual_bound)
    dual = -raw_dual if info.valid and np.isfinite(raw_dual) else np.nan
    if np.isfinite(dual) and has_available and dual < value - TOL:
        raise AssertionError("Upper bound is below the validated feasible value")

    method = f"HiGHS-{budget:g}s-{label}"
    end_to_end = (package["graph_time"] + package["matrix_time"] + heuristic_time +
                  mapping_time + lp_time + model_load_time + submission_time + solver_wall_time)
    summary = {
        "Scenario": scenario["scenario_id"], "Method": method, "Start Method": label,
        "Warm Start Submitted": submitted, "Warm Start Value": start_value,
        "Solver Has Incumbent": solver_has, "Solver Incumbent Value": solver_value,
        "Has Incumbent": has_available, "Available Plan Source": available_source,
        "Has Positive Plan": bool(has_available and value > 1e-10),
        "Optimal Certified": certified, "Primal Bound": value,
        "Solver Primal Bound": solver_value, "Dual Bound": dual,
        "MIP Gap": float(info.mip_gap) if info.valid else np.nan,
        "MIP Nodes": int(info.mip_node_count),
        "Variables": len(package["c"]), "Constraints": package["A"].shape[0],
        "Nonzeros": package["A"].nnz,
        "Solver Message": highs.modelStatusToString(model_status),
        "Solver Status Code": int(model_status), "Time Limit sec": float(budget),
        "Heuristic Time sec": heuristic_time, "Warm Start Mapping sec": mapping_time,
        "Native Model Conversion sec": lp_time, "Model Load sec": model_load_time,
        "Start Submission sec": submission_time,
        "Solver Time sec": solver_wall_time, "HiGHS Internal Time sec": highs.getRunTime(),
        "Runtime sec": solver_wall_time, "Graph Time sec": package["graph_time"],
        "Matrix Time sec": package["matrix_time"],
        "Previous Solver Time sec": previous_solver_time,
        "Previous End-to-end sec": previous_end_to_end,
        "Previous Solver Budget sec": previous_budget,
        "Sequential Solver Budget sec": previous_budget + float(budget),
        "End-to-end sec": end_to_end,
        "Total Sequential End-to-end sec": end_to_end + previous_end_to_end,
        "HiGHS Version": highs.version(), "Threads": threads, "Random Seed": solver_seed,
        "Source Commit": SOURCE_COMMIT, "Objective Version": OBJECTIVE_VERSION,
    }
    result = SimpleNamespace(x=vector, fun=-value if has_available else np.nan,
                             success=certified, status=model_status,
                             message=summary["Solver Message"], mip_dual_bound=raw_dual,
                             mip_gap=summary["MIP Gap"], mip_node_count=info.mip_node_count)
    return result, summary


def read_replay(args, scenario):
    if not args.replay_logs:
        return None
    metrics = pd.read_csv(args.replay_metrics)
    match = metrics[metrics["Scenario"].eq(scenario["scenario_id"]) &
                    metrics["Method"].eq(args.replay_method)]
    if len(match) != 1 or str(match.iloc[0]["Has Incumbent"]).lower() != "true":
        raise ValueError("Replay requires one recorded incumbent for this exact scenario")
    row = match.iloc[0]
    if str(row["Optimal Certified"]).lower() == "true" and not args.allow_oracle_replay:
        raise ValueError("An optimal replay is an oracle check; use --allow-oracle-replay explicitly")
    logs = pd.read_csv(args.replay_logs)
    frame = logs[logs["Scenario"].eq(scenario["scenario_id"]) &
                 logs["Method"].eq(args.replay_method)].copy()
    return frame, row


def run_instance(kernel, args, experiment, tasks, seed, directory):
    scenario = kernel["create_scenario"](tasks, seed)
    kernel["validate_scenario_structure"](scenario)
    kernel["validate_temporal_satisfaction"](scenario)
    fingerprint = scenario_fingerprint(scenario)
    rows, route_frames, greedy_logs, greedy_times = [], [], {}, {}
    for label, rule in (("GBStart", "benefit"), ("GDStart", "density_operational")):
        frame, summary = kernel["solve_greedy_ttap"](scenario, rule=rule)
        kernel["validate_solution_consistency"](scenario, frame, summary["Method"])
        row = kernel["compute_ttap_metrics"](scenario, frame, summary)
        row.update({"Configuration": experiment, "Start Method": "Heuristic",
                    "Source Commit": SOURCE_COMMIT, "Objective Version": OBJECTIVE_VERSION,
                    "Scenario SHA256": fingerprint})
        rows.append(row)
        route_frames.append(frame)
        greedy_logs[label], greedy_times[label] = frame, float(summary["Runtime sec"])
    print(f"{scenario['scenario_id']}: building state-arc MILP", flush=True)
    package = kernel["prepare_exact_problem"](scenario)
    print(f"  {len(package['c']):,} variables; {package['A'].shape[0]:,} rows", flush=True)

    replay = read_replay(args, scenario)
    if replay is not None:
        greedy_logs["ReplayStart"] = replay[0]
        greedy_times["ReplayStart"] = 0.0
    starts, mapping_times = {}, {}
    # Map separately so each treatment is charged its actual setup cost.
    for label, frame in greedy_logs.items():
        t0 = time.perf_counter()
        starts[label] = logs_to_start_vectors(kernel, scenario, package, {label: frame})[label]
        mapping_times[label] = time.perf_counter() - t0
    if replay is not None and not math.isclose(
        -float(package["c"] @ starts["ReplayStart"]),
        float(replay[1]["Normalized Mission Value"]), abs_tol=1e-8
    ):
        raise ValueError("Replay scenario/objective does not match the recorded run")
    signature = model_fingerprint(package)
    t0 = time.perf_counter()
    lp = to_highs_lp(package)
    lp_time = time.perf_counter() - t0

    directory.mkdir(parents=True, exist_ok=True)
    treatments = ["Cold", "GBStart", "GDStart"] + (["ReplayStart"] if replay else [])
    # Rotate execution order across seeds; all treatments still receive the same settings.
    offset = seed % len(treatments)
    treatments = treatments[offset:] + treatments[:offset]
    for budget in args.budgets:
        for label in treatments:
            result, summary = solve_one(
                scenario, package, lp, budget, label, start=starts.get(label),
                heuristic_time=greedy_times.get(label, 0.0),
                mapping_time=mapping_times.get(label, 0.0), lp_time=lp_time,
                threads=args.threads, solver_seed=args.solver_seed,
                mip_rel_gap=args.mip_rel_gap, verbose=args.verbose,
                previous_solver_time=float(replay[1]["Solver Time sec"]) if label == "ReplayStart" else 0.0,
                previous_budget=float(replay[1]["Time Limit sec"]) if label == "ReplayStart" else 0.0,
                previous_end_to_end=float(replay[1]["End-to-end sec"]) if label == "ReplayStart" else 0.0)
            kernel["validate_exact_incumbent"](result, summary, package)
            frame = kernel["decode_exact_solution"](scenario, result, package["all_arcs"], summary["Method"])
            kernel["validate_solution_consistency"](scenario, frame, summary["Method"])
            row = kernel["compute_ttap_metrics"](scenario, frame, summary)
            row.update({"Configuration": experiment, "Scenario SHA256": fingerprint,
                        "Model SHA256": signature})
            if result.x is not None and not math.isclose(
                row["Normalized Mission Value"], -result.fun, abs_tol=TOL):
                raise AssertionError("Decoded routes do not reproduce the incumbent objective")
            rows.append(row)
            route_frames.append(frame)
            # Sparse selected columns are for auditing/replaying THIS model only.
            if result.x is not None:
                np.savez_compressed(directory / f"{summary['Method']}.npz",
                                    selected=np.flatnonzero(result.x > 0.5),
                                    model_sha256=np.array(signature),
                                    scenario_sha256=np.array(fingerprint))
            pd.DataFrame(rows).to_csv(directory / "metrics_partial.csv", index=False)
            print(f"  {summary['Method']}: value={row['Normalized Mission Value']:.6g}; "
                  f"solver incumbent={summary['Solver Has Incumbent']}; "
                  f"certified={summary['Optimal Certified']}; "
                  f"source={summary['Available Plan Source']}", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(directory / "metrics.csv", index=False)
    pd.concat(route_frames, ignore_index=True).to_csv(directory / "routes.csv", index=False)
    del package, starts, lp
    gc.collect()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=["A", "B"], default=["A"])
    parser.add_argument("--tasks", nargs="+", type=int, default=[20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--budgets", nargs="+", type=float, default=[1, 5, 15, 60])
    parser.add_argument("--output", type=Path, required=True,
                        help="New run directory; it is never silently overwritten")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-6)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--replay-logs", type=Path)
    parser.add_argument("--replay-metrics", type=Path)
    parser.add_argument("--replay-method")
    parser.add_argument("--allow-oracle-replay", action="store_true")
    args = parser.parse_args()
    if bool(args.replay_logs) != bool(args.replay_metrics and args.replay_method):
        parser.error("Replay requires --replay-logs, --replay-metrics and --replay-method together")
    if args.replay_logs and len(args.experiments) != 1:
        parser.error("Replay one experiment at a time; A routes are not portable to B")
    if any(t not in [20, 30, 40, 50, 60] for t in args.tasks):
        parser.error("Use the original task counts: 20, 30, 40, 50, 60")
    if any(s not in range(1, 31) for s in args.seeds) or any(b <= 0 for b in args.budgets):
        parser.error("Seeds must be 1..30; budgets must be positive")
    if len(set(args.budgets)) != len(args.budgets):
        parser.error("Duplicate time budgets")
    if (len(set(args.experiments)) != len(args.experiments) or len(set(args.tasks)) != len(args.tasks)
        or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Duplicate experiments, task counts, or seeds")
    settings = {k: v for k, v in vars(args).items() if k not in {"resume", "output"}}
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}
    manifest = {"settings": settings, "source_commit": SOURCE_COMMIT,
                "objective_version": OBJECTIVE_VERSION, "highs": highspy.Highs().version(),
                "python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    source_hash = hashlib.sha256()
    for path in sorted(ROOT.glob("frozen_*.py")):
        source_hash.update(path.read_bytes())
    manifest["frozen_source_sha256"] = source_hash.hexdigest()
    manifest["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume:
            parser.error("Output directory is nonempty; choose a new directory or --resume")
        previous = json.loads((args.output / "manifest.json").read_text())
        if previous != manifest:
            parser.error("Cannot resume with changed settings, solver, or frozen sources")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs = []
    for experiment in args.experiments:
        kernel = load_kernel(experiment)
        for tasks in args.tasks:
            for seed in args.seeds:
                directory = args.output / f"{experiment}_T{tasks}_S{seed}"
                complete = directory / "metrics.csv"
                if args.resume and complete.exists():
                    outputs.append(pd.read_csv(complete))
                    continue
                outputs.append(run_instance(kernel, args, experiment, tasks, seed, directory))
                pd.concat(outputs, ignore_index=True).to_csv(args.output / "metrics_partial.csv", index=False)
    pd.concat(outputs, ignore_index=True).to_csv(args.output / "metrics.csv", index=False)
    print(f"Completed. Results: {args.output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
