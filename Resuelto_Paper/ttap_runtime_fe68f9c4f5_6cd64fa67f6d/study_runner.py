"""Reproducible A/B warm-start study, with optional new long-budget references.

Warm jobs are independent restarts. Reference jobs run afterwards and never
seed a warm job. The legacy runner remains available for historical replay.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import gc
import hashlib
import json
import math
import platform
import sys
import time

import highspy
import numpy as np
import pandas as pd
import scipy
import matplotlib

import ttap_warm_start as native
from objective_profiles import ObjectiveSpec, apply_objective, compute_metrics


ROOT = Path(__file__).resolve().parent
TREATMENTS = ("Cold", "GBStart", "GDStart")
PROTOCOL = "TTAP-greedy-MIP-starts-v2"


@dataclass(frozen=True)
class StudyConfig:
    experiment: str
    tasks: tuple = (20,)
    seeds: tuple = (1,)
    budgets: tuple = (1., 5., 15., 60.)
    reference_budgets: tuple = ()
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    threads: int = 1
    solver_seed: int = 0
    mip_rel_gap: float = 1e-6
    trace: bool = True

    def __post_init__(self):
        if self.experiment not in ("A", "B"):
            raise ValueError("experiment must be A or B")
        if not self.tasks or any(t not in (20, 30, 40, 50, 60) for t in self.tasks):
            raise ValueError("Use original task counts: 20, 30, 40, 50, 60")
        if not self.seeds or any(s not in range(1, 31) for s in self.seeds):
            raise ValueError("Use original seeds 1..30")
        for values in (self.tasks, self.seeds, self.budgets, self.reference_budgets):
            if len(set(values)) != len(values):
                raise ValueError("Duplicate tasks, seeds or budgets")
        if not self.budgets or any(not math.isfinite(float(b)) or b <= 0
                                   for b in self.budgets + self.reference_budgets):
            raise ValueError("Solver budgets must be finite and positive")
        if self.threads < 1 or not 0 <= self.mip_rel_gap < 1:
            raise ValueError("Invalid thread count or relative tolerance")

    def as_dict(self):
        result = asdict(self)
        result["objective"] = self.objective.as_dict()
        return json.loads(json.dumps(result))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def full_cohort(self):
        return (set(self.tasks) == {20, 30, 40, 50, 60}
                and set(self.seeds) == set(range(1, 31))
                and set(self.budgets) == {1., 5., 15., 60.})

    def workload(self):
        cases = len(self.tasks) * len(self.seeds)
        return {"instances": cases, "warm_solver_runs": cases * 3 * len(self.budgets),
                "reference_solver_runs": cases * len(self.reference_budgets),
                "nominal_warm_solver_hours": cases * 3 * sum(self.budgets) / 3600,
                "nominal_reference_solver_hours": cases * sum(self.reference_budgets) / 3600,
                "timing_note": "Allowances only: setup and time-limit overruns are additional; optimal runs may stop early."}


def configuration(experiment, mode="pilot", objective=None):
    """Pilot first; full preserves the original five sizes and 30 seeds."""
    objective = objective or ObjectiveSpec()
    if mode == "smoke":
        tasks, seeds, budgets = (20,), (1,), (1.,)
    elif mode == "pilot":
        tasks, seeds, budgets = (20,), (1,), (1., 5., 15., 60.)
    elif mode == "screen":
        tasks, seeds, budgets = (20, 40, 60), (1, 2, 3), (1., 5., 15., 60.)
    elif mode == "full":
        tasks, seeds, budgets = (20, 30, 40, 50, 60), tuple(range(1, 31)), (1., 5., 15., 60.)
    else:
        raise ValueError("mode must be smoke, pilot, screen or full")
    references = (600.,) if experiment == "A" else (600., 1800.)
    return StudyConfig(experiment, tasks, seeds, budgets, references, objective)


def atomic_csv(frame, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def source_signatures():
    names = ["frozen_generator_A.py", "frozen_generator_B.py", "frozen_ttap_core.py",
             "ttap_warm_start.py", "objective_profiles.py", "study_runner.py"]
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


def manifest_for(config):
    return {"protocol": PROTOCOL, "config": config.as_dict(),
            "config_sha256": config.fingerprint, "base_commit": native.SOURCE_COMMIT,
            "source_files_sha256": source_signatures(),
            "python": sys.version.split()[0], "highs": highspy.Highs().version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "machine": platform.machine(),
            "cohort": "full" if config.full_cohort else "pilot"}


def job_key(stage, label, budget):
    return f"{stage}:{label}:{float(budget):g}"


def scenario_rows(scenario, config):
    rows = []
    for i, task in scenario["tasks"].items():
        rows.append({"Configuration": config.experiment, "Scenario": scenario["scenario_id"],
                     "Task ID": i, "Node": scenario["task_to_node"][i],
                     "Medical": task["med"], "Personnel": task["pers"], "Cargo kg": task["load"],
                     "Task class": task["priority_class"], "Potential value": task["potential_value"],
                     "Medical value component": task["value_components"]["medical"],
                     "Personnel value component": task["value_components"]["personnel"],
                     "Cargo value component": task["value_components"]["cargo"],
                     "Objective SHA256": config.objective.fingerprint})
    return pd.DataFrame(rows)


def run_case(config, tasks, seed, folder, phase, resume=True, verbose=False):
    folder.mkdir(parents=True, exist_ok=True)
    metrics_path = folder / "metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if resume and metrics_path.exists() else []
    if rows and any(str(r.get("Objective SHA256")) != config.objective.fingerprint for r in rows):
        raise ValueError("Checkpoint objective does not match the requested objective")
    jobs = ([("Warm", m, float(b)) for b in config.budgets for m in TREATMENTS]
            if phase == "warm" else [("Reference", "ColdReference", float(b)) for b in config.reference_budgets])
    completed = {r.get("Job ID") for r in rows}
    if all(job_key(*job) in completed for job in jobs):
        print(f"{config.experiment}/T{tasks}/S{seed}: {phase} already completed", flush=True)
        return pd.DataFrame(rows)

    kernel = native.load_kernel(config.experiment)
    scenario = kernel["create_scenario"](tasks, seed)
    kernel["validate_scenario_structure"](scenario)
    apply_objective(scenario, config.objective)
    kernel["validate_temporal_satisfaction"](scenario)
    fingerprint = native.scenario_fingerprint(scenario)
    if rows and any(str(r.get("Scenario SHA256")) != fingerprint for r in rows):
        raise ValueError("Checkpoint scenario does not match the current instance")
    scenario_rows(scenario, config).to_csv(folder / "task_values.csv", index=False)
    trace_path, routes_path = folder / "trace.csv", folder / "routes.csv"
    traces = pd.read_csv(trace_path).to_dict("records") if resume and trace_path.exists() else []
    traces = [e for e in traces if e.get("Job ID") in completed]
    committed_methods = {r.get("Method") for r in rows}
    route_frames = []
    if resume and routes_path.exists():
        saved_routes = pd.read_csv(routes_path)
        route_frames = [saved_routes[saved_routes["Method"].isin(committed_methods)]]
    logs, greedy_times = {}, {}
    for label, rule in (("GBStart", "benefit"), ("GDStart", "density_operational")):
        frame, summary = kernel["solve_greedy_ttap"](scenario, rule=rule)
        kernel["validate_solution_consistency"](scenario, frame, summary["Method"])
        logs[label], greedy_times[label] = frame, float(summary["Runtime sec"])
        jid = job_key("Heuristic", label, 0)
        if jid not in completed:
            row = compute_metrics(kernel, scenario, frame, summary, config.objective)
            row.update({"Configuration": config.experiment, "Study Stage": "Heuristic",
                        "Start Method": label, "Job ID": jid, "Scenario SHA256": fingerprint,
                        "Source Commit": native.SOURCE_COMMIT, "Protocol": PROTOCOL})
            rows.append(row)
            route_frames.append(frame)

    print(f"{scenario['scenario_id']}: build {config.objective.mode} MILP ({phase})", flush=True)
    package = kernel["prepare_exact_problem"](scenario)
    signature = native.model_fingerprint(package)
    print(f"  {len(package['c']):,} variables; {package['A'].shape[0]:,} constraints", flush=True)
    starts, mapping_times = {}, {}
    if phase == "warm":
        for label, frame in logs.items():
            t0 = time.perf_counter()
            starts[label] = native.logs_to_start_vectors(kernel, scenario, package, {label: frame})[label]
            mapping_times[label] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lp = native.to_highs_lp(package)
    lp_time = time.perf_counter() - t0

    if phase == "warm":
        order = list(TREATMENTS)
        offset = seed % len(order)
        order = order[offset:] + order[:offset]
        jobs = [("Warm", m, float(b)) for b in config.budgets for m in order]
    for stage, label, budget in jobs:
        jid = job_key(stage, label, budget)
        if jid in completed:
            continue
        events = [] if config.trace else None
        result, summary = native.solve_one(
            scenario, package, lp, budget, label, start=starts.get(label),
            heuristic_time=greedy_times.get(label, 0.) if label in starts else 0.,
            mapping_time=mapping_times.get(label, 0.), lp_time=lp_time,
            threads=config.threads, solver_seed=config.solver_seed,
            mip_rel_gap=config.mip_rel_gap, verbose=verbose, trace=events)
        summary.update({"Objective Version": config.objective.version,
                        "Objective SHA256": config.objective.fingerprint,
                        "Study Stage": stage, "Job ID": jid,
                        "Protocol": PROTOCOL, "MIP Relative Tolerance": config.mip_rel_gap})
        native.validate_vector(package, result.x) if result.x is not None else None
        kernel["validate_exact_incumbent"](result, summary, package)
        frame = kernel["decode_exact_solution"](scenario, result, package["all_arcs"], summary["Method"])
        kernel["validate_solution_consistency"](scenario, frame, summary["Method"])
        row = compute_metrics(kernel, scenario, frame, summary, config.objective)
        row.update({"Configuration": config.experiment, "Scenario SHA256": fingerprint,
                    "Model SHA256": signature})
        if result.x is not None and not math.isclose(row["Normalized Mission Value"], -result.fun, abs_tol=1e-6):
            raise AssertionError("Decoded objective does not match the validated solution")
        if events is not None:
            observed = [e["Elapsed solve wall sec"] for e in events
                        if e["Event"] == "HiGHS improving incumbent"]
            improvements = [e["Elapsed solve wall sec"] for e in events
                            if e["Event"] == "HiGHS improving incumbent"
                            and math.isfinite(summary["Warm Start Value"])
                            and e["Solver Incumbent Value"] > summary["Warm Start Value"] + 1e-6]
            row["First Observed Solver Incumbent sec"] = min(observed) if observed else float("nan")
            row["First Improvement Over Start sec"] = min(improvements) if improvements else float("nan")
            for event in events:
                event.update({"Configuration": config.experiment, "Study Stage": stage,
                              "Objective SHA256": config.objective.fingerprint, "Job ID": jid})
            traces.extend(events)
        rows.append(row)
        route_frames.append(frame)
        if result.x is not None:
            np.savez_compressed(folder / f"{summary['Method']}.npz",
                                selected=np.flatnonzero(result.x > .5),
                                model_sha256=np.array(signature), scenario_sha256=np.array(fingerprint))
        # The metrics checkpoint is the completion marker and is replaced last.
        atomic_csv(pd.concat(route_frames, ignore_index=True), routes_path)
        if traces:
            atomic_csv(pd.DataFrame(traces), trace_path)
        atomic_csv(pd.DataFrame(rows), metrics_path)
        print(f"  {summary['Method']}: value={row['Normalized Mission Value']:.6f}; "
              f"solver plan={summary['Solver Has Incumbent']}; certified={summary['Optimal Certified']}", flush=True)
    del package, lp, starts
    gc.collect()
    return pd.DataFrame(rows)


def run_study(config, output, phase="warm", resume=True, verbose=False):
    if phase not in ("warm", "reference"):
        raise ValueError("phase must be warm or reference")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    manifest = manifest_for(config)
    if path.exists():
        if not resume:
            raise ValueError("Output already exists; use resume=True or a new directory")
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Cannot resume with different protocol, settings, sources or environment")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output directory without a matching manifest")
    path.write_text(json.dumps(manifest, indent=2))
    for tasks in config.tasks:
        for seed in config.seeds:
            run_case(config, tasks, seed, output / f"{config.experiment}_T{tasks}_S{seed}",
                     phase, resume, verbose)
            files = sorted(output.glob("*/metrics.csv"))
            atomic_csv(pd.concat([pd.read_csv(f) for f in files], ignore_index=True), output / "metrics.csv")
    for name in ("trace.csv", "routes.csv", "task_values.csv"):
        files = sorted(output.glob(f"*/{name}"))
        if files:
            atomic_csv(pd.concat([pd.read_csv(f) for f in files], ignore_index=True), output / name)
    return pd.read_csv(output / "metrics.csv")
