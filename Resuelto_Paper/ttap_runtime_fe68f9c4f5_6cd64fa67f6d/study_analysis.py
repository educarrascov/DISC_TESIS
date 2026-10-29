"""Budget quality and attribution for the explicit-objective TTAP study.

Unknown optima remain unknown. A missing returned incumbent is never silently
treated as a recorded zero-valued solution. An idle=0 available-plan statistic
is exported separately. All gaps are computed per instance before averaging.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from objective_profiles import ObjectiveSpec
from figure5_and_analysis import booleans


KEYS = ["Configuration", "Scenario", "Tasks", "Seed", "Objective SHA256"]
PAIR = KEYS + ["Time Limit sec"]
METHODS = ("Cold", "GBStart", "GDStart")
COLORS = {"Cold": "#555555", "GBStart": "#D55E00", "GDStart": "#009E73"}
LABELS = {"Cold": "HiGHS without start", "GBStart": "HiGHS + Greedy Benefit",
          "GDStart": "HiGHS + Greedy Density"}


def prepare_study(raw, spec, legacy_references=None, require_full=False):
    data = raw.copy()
    if data.empty or not data["Objective SHA256"].eq(spec.fingerprint).all():
        raise ValueError("Mixed or incorrect objective fingerprints")
    for col in ("Has Incumbent", "Solver Has Incumbent", "Optimal Certified"):
        if col in data:
            data[col] = booleans(data[col])
    solver = data[data["Study Stage"].isin(["Warm", "Reference"])].copy()
    if solver.duplicated(PAIR + ["Start Method", "Study Stage"]).any():
        raise ValueError("Duplicate solver observations")
    solver["Available value"] = pd.to_numeric(solver["Normalized Mission Value"]).fillna(0.)
    solver["Recorded solver value"] = pd.to_numeric(solver["Solver Incumbent Value"], errors="coerce").where(
        solver["Solver Has Incumbent"])
    if solver.loc[solver["Has Incumbent"], "Normalized Mission Value"].isna().any():
        raise ValueError("Available incumbent without a finite value")
    if solver.loc[solver["Solver Has Incumbent"], "Recorded solver value"].isna().any():
        raise ValueError("Solver incumbent without a finite value")
    if (solver["Optimal Certified"] & ~solver["Solver Has Incumbent"]).any():
        raise ValueError("Certified status without a solver incumbent")
    for key, group in solver.groupby(KEYS, dropna=False):
        for column in ("Model SHA256", "Scenario SHA256"):
            if group[column].isna().any() or group[column].nunique() != 1:
                raise ValueError(f"Changed {column} between reference and warm jobs for {key}")
    warm = solver[solver["Study Stage"].eq("Warm")].copy()
    for key, group in warm.groupby(PAIR, dropna=False):
        if set(group["Start Method"]) != set(METHODS) or len(group) != 3:
            raise ValueError(f"Incomplete paired treatments for {key}")
        for column in ("Model SHA256", "Scenario SHA256", "HiGHS Version", "Threads", "Random Seed"):
            if group[column].isna().any() or group[column].nunique() != 1:
                raise ValueError(f"Unmatched {column} for {key}")
    for column in ("HiGHS Version", "Threads", "Random Seed"):
        if warm[column].nunique() != 1:
            raise ValueError(f"Do not pool different {column} settings")
    if require_full:
        expected = {(t, s, b, m) for t in (20, 30, 40, 50, 60) for s in range(1, 31)
                    for b in (1., 5., 15., 60.) for m in METHODS}
        for config, group in warm.groupby("Configuration"):
            actual = set(map(tuple, group[["Tasks", "Seed", "Time Limit sec", "Start Method"]].to_numpy()))
            if actual != expected:
                raise ValueError(f"{config}: not the full 30-replicate paper cohort")

    certified = solver[solver["Optimal Certified"]].copy()
    references = certified.groupby(KEYS, as_index=False).agg(
        **{"Certified optimum": ("Normalized Mission Value", "max")})
    if legacy_references is not None:
        if not spec.historical_compatible:
            raise ValueError("Historical optima are invalid for a changed objective")
        legacy = legacy_references.copy()
        if not legacy["Objective SHA256"].eq(spec.fingerprint).all():
            raise ValueError("Historical reference fingerprint mismatch")
        references = pd.concat([references, legacy[KEYS + ["Certified optimum"]]], ignore_index=True)
        references = references.groupby(KEYS, as_index=False)["Certified optimum"].max()
    # Return/count only reference instances belonging to this run's cohort.
    references = references.merge(solver[KEYS].drop_duplicates(), on=KEYS,
                                  how="inner", validate="one_to_one")
    solver = solver.merge(references, on=KEYS, how="left", validate="many_to_one")
    if (solver["Available value"] > solver["Certified optimum"] + 1e-6).any():
        raise ValueError("Available value exceeds certified reference: inspect objective/model identity")
    gap = 100 * (solver["Certified optimum"] - solver["Available value"]) / solver["Certified optimum"]
    solver["Available shortfall pct"] = gap.clip(lower=0)
    solver["Solver shortfall pct"] = (100 * (solver["Certified optimum"] - solver["Recorded solver value"])
                                      / solver["Certified optimum"]).clip(lower=0)
    warm = solver[solver["Study Stage"].eq("Warm")].copy()
    heur = data[data["Study Stage"].eq("Heuristic")].copy()
    if heur.duplicated(KEYS + ["Start Method"]).any():
        raise ValueError("Duplicate greedy baselines")
    baselines = heur.pivot(index=KEYS, columns="Start Method", values="Normalized Mission Value").reset_index()
    if not {"GBStart", "GDStart"}.issubset(baselines.columns):
        raise ValueError("Both greedy baselines are required")
    baselines = baselines.rename(columns={"GBStart": "GB value", "GDStart": "GD value"})
    warm = warm.merge(baselines, on=KEYS, how="left", validate="many_to_one")
    if warm[["GB value", "GD value"]].isna().any().any():
        raise ValueError("Missing matched greedy plan")
    cold = warm[warm["Start Method"].eq("Cold")][PAIR + ["Available value"]].rename(
        columns={"Available value": "Cold available value"})
    gains = warm[warm["Start Method"].ne("Cold")].merge(cold, on=PAIR, validate="many_to_one")
    gains["Initial greedy value"] = np.where(gains["Start Method"].eq("GBStart"), gains["GB value"], gains["GD value"])
    if not np.allclose(gains["Initial greedy value"], gains["Warm Start Value"], atol=1e-8, rtol=0):
        raise ValueError("Recorded greedy differs from the supplied start")
    if (gains["Available value"] < gains["Initial greedy value"] - 1e-8).any():
        raise ValueError("A validated initial plan was lost")
    gains["Gain over greedy"] = gains["Available value"] - gains["Initial greedy value"]
    gains["Injection gain"] = gains["Available value"] - np.maximum(
        gains["Cold available value"], gains["Initial greedy value"])
    gains["Improved greedy"] = gains["Gain over greedy"].gt(1e-6)
    gains["Beat cold plus greedy"] = gains["Injection gain"].gt(1e-6)
    return solver, warm, heur, gains, references


def summaries(warm, gains, by_task=True):
    keys = ["Configuration"] + (["Tasks"] if by_task else []) + ["Start Method", "Time Limit sec"]
    table = warm.groupby(keys, as_index=False).agg(
        n=("Scenario", "size"), solver_incumbents=("Solver Has Incumbent", "sum"),
        available_plans=("Has Incumbent", "sum"), certified=("Optimal Certified", "sum"),
        mean_solver_incumbent=("Recorded solver value", "mean"),
        median_solver_incumbent=("Recorded solver value", "median"),
        mean_available_value=("Available value", "mean"),
        n_known_optima=("Certified optimum", "count"),
        n_known_incumbent_gaps=("Solver shortfall pct", "count"),
        mean_solver_shortfall_pct=("Solver shortfall pct", "mean"),
        median_solver_shortfall_pct=("Solver shortfall pct", "median"),
        mean_available_shortfall_pct=("Available shortfall pct", "mean"),
        mean_measured_seconds=("End-to-end sec", "mean"))
    for name in ("solver_incumbents", "available_plans", "certified"):
        table[name + "_pct"] = 100 * table[name] / table["n"]
    attr = gains.groupby(keys, as_index=False).agg(
        n=("Scenario", "size"), mean_gain_over_greedy=("Gain over greedy", "mean"),
        median_gain_over_greedy=("Gain over greedy", "median"),
        mean_injection_gain=("Injection gain", "mean"), median_injection_gain=("Injection gain", "median"),
        improved_greedy=("Improved greedy", "sum"), beat_cold_plus_greedy=("Beat cold plus greedy", "sum"))
    return table, attr


def save_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def trace_example(run_directory, solver, output):
    path = Path(run_directory) / "trace.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data[data["Study Stage"].eq("Warm")]
    if data.empty:
        return
    first = data.sort_values(["Tasks", "Seed"]).iloc[0]
    scenario, budget = first["Scenario"], data["Budget sec"].max()
    data = data[data["Scenario"].eq(scenario) & data["Budget sec"].eq(budget)]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    fig.subplots_adjust(left=.11, right=.97, top=.74, bottom=.30)
    for m in METHODS:
        rows = data[data["Start Method"].eq(m)].sort_values("Elapsed solve wall sec", kind="stable")
        if not rows.empty:
            ax.step(rows["Elapsed solve wall sec"], rows["Available Value"], where="post",
                    color=COLORS[m], label=LABELS[m])
            ax.scatter([0], [rows.iloc[0]["Available Value"]], color=COLORS[m], marker="s", s=20)
    ref = solver.loc[solver["Scenario"].eq(scenario), "Certified optimum"].dropna().unique()
    if len(ref) == 1:
        ax.axhline(ref[0], color="#777777", linestyle=":", label="Certified reference")
    ax.set_ylim(-.03, 1.05)
    ax.set_xlabel("Elapsed solve wall time (s)")
    ax.set_ylabel("Best available normalized benefit")
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    fig.suptitle(f"Illustrative single-instance trace: {scenario}, {budget:g}s allowance", fontsize=10, y=.98)
    fig.legend(loc="upper center", bbox_to_anchor=(.55, .91), ncol=2, frameon=False, fontsize=7)
    fig.text(.11, .035, "Each curve is one actual search, recorded by native callbacks. Squares credit the initial plan or idle=0.\n"
             "The solve clock starts after preparation; an external initial plan is not a HiGHS discovery.\n"
             "This illustrates one instance and does not replace the paired cohort comparison.", fontsize=7)
    save_figure(fig, output, "Figure_WarmStart_trace_example")


def axes_budget(ax, budgets):
    ax.set_xscale("log")
    ax.set_xticks(budgets, [f"{v:g}" for v in budgets])
    ax.set_xlabel("Solver allowance (s)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)


def plot_report(solver, warm, heur, gains, output, is_full):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    overall, attr = summaries(warm, gains, by_task=False)
    budgets = sorted(warm["Time Limit sec"].unique())
    config = warm["Configuration"].iloc[0]
    n = warm[KEYS].drop_duplicates().shape[0]
    title = f"Experiment {config}: " + ("full cohort" if is_full else f"functional pilot, n={n}; not paper results")
    handles = [Line2D([], [], color=COLORS[m], marker="o", label=LABELS[m]) for m in METHODS]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    fig.subplots_adjust(left=.11, right=.97, top=.80, bottom=.23, hspace=.58, wspace=.28)
    definitions = [("solver_incumbents_pct", "Solver-returned incumbent (%)", 100),
                   ("mean_solver_incumbent", "Mean recorded solver benefit", 1),
                   ("mean_solver_shortfall_pct", "Mean shortfall to known optimum (%)", 100),
                   ("certified_pct", "Optimality certified (%)", 100)]
    for ax, (column, label, maximum) in zip(axes.flat, definitions):
        for m in METHODS:
            rows = overall[overall["Start Method"].eq(m)].sort_values("Time Limit sec")
            ax.plot(rows["Time Limit sec"], rows[column], color=COLORS[m], marker="o", markersize=4)
        if not overall[column].notna().any():
            ax.text(.5, .5, "No certified references yet" if "shortfall" in column else "No returned incumbent",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
        axes_budget(ax, budgets)
        ax.set_ylim(-.03 * maximum, 1.06 * maximum)
        ax.set_title(label, fontsize=9)
    fig.suptitle(title, fontsize=11, y=.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .94), ncol=2, frameon=False)
    fig.text(.11, .045, "Means/gaps exclude missing solver incumbents; sample counts are exported. Known-optimum gaps use\n"
             "certified references only. Initial starts contribute to returned-plan availability; this is not discovery.\n"
             "Each allowance is an independent restart. Solver allowances exclude model/start preparation.", fontsize=7)
    save_figure(fig, output, "Figure_WarmStart_quality")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    fig.subplots_adjust(left=.12, right=.97, top=.74, bottom=.32, wspace=.32)
    for ax, column, ylabel in zip(axes, ["mean_gain_over_greedy", "mean_injection_gain"],
                                  ["Benefit gained over initial greedy", "Gain over cold + same greedy fallback"]):
        for m in ("GBStart", "GDStart"):
            rows = attr[attr["Start Method"].eq(m)].sort_values("Time Limit sec")
            ax.plot(rows["Time Limit sec"], rows[column], color=COLORS[m], marker="o")
        ax.axhline(0, color="#777777", linestyle=":", linewidth=1)
        axes_budget(ax, budgets)
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10, y=.98)
    fig.legend(handles=handles[1:], loc="upper center", bbox_to_anchor=(.55, .90), ncol=2, frameon=False)
    fig.text(.12, .035, "Right: warm value minus max(new cold value, same initial greedy). Negative gains are retained.\n"
             "Both comparisons use paired instances and equal solver allowances; setup is reported separately.", fontsize=7)
    save_figure(fig, output, "Figure_WarmStart_attribution")

    fig, axes = plt.subplots(1, 5, figsize=(9, 3.1), sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.33, wspace=.16)
    for ax, count in zip(axes, (20, 30, 40, 50, 60)):
        for m in METHODS:
            rows = solver[solver["Tasks"].eq(count) & solver["Start Method"].eq(m)]
            if m == "Cold":
                rows = solver[solver["Tasks"].eq(count) & solver["Start Method"].isin(["Cold", "ColdReference"])]
            curve = rows.groupby("Time Limit sec")["Optimal Certified"].mean() * 100
            if not curve.empty:
                ax.plot(curve.index, curve, color=COLORS[m], marker="o", markersize=3)
        b = sorted(solver["Time Limit sec"].unique())
        axes_budget(ax, b)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax.set_ylim(-3, 105)
        ax.set_title(f"{count} tasks")
        if not solver["Tasks"].eq(count).any():
            ax.text(.5, .5, "Not run", ha="center", transform=ax.transAxes, fontsize=7)
    axes[0].set_ylabel("Certified runs (%)")
    fig.suptitle(title, fontsize=10, y=.98)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .90), ncol=3, frameon=False, fontsize=7)
    fig.text(.08, .025, "Original Figure-5 style. Each point uses the generated seed cohort for that task count.\n"
             "Only cold reference jobs extend beyond the four short warm-start budgets; absence means not run.", fontsize=7)
    save_figure(fig, output, "Figure_WarmStart_certification_by_tasks")


def export_report(run_directory, use_legacy=True, require_full=False):
    run_directory = Path(run_directory)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    value = manifest["config"]["objective"]
    spec = ObjectiveSpec(value["mode"], value["weights"], value["references"])
    legacy = None
    legacy_path = Path(__file__).resolve().parent / "historical_reference_values.csv"
    if use_legacy and spec.historical_compatible and legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
    elif use_legacy and spec.historical_compatible:
        from public_sources import historical_references
        legacy = historical_references(spec)
    solver, warm, heur, gains, refs = prepare_study(pd.read_csv(run_directory / "metrics.csv"),
                                                 spec, legacy, require_full)
    output = run_directory / "analysis"
    output.mkdir(exist_ok=True)
    solver.to_csv(output / "solver_paired_rows.csv", index=False)
    gains.to_csv(output / "paired_warm_gains.csv", index=False)
    refs.to_csv(output / "certified_reference_values.csv", index=False)
    for by_task, tag in ((True, "by_task"), (False, "pooled")):
        table, attr = summaries(warm, gains, by_task)
        table.to_csv(output / f"budget_quality_{tag}.csv", index=False)
        attr.to_csv(output / f"warm_attribution_{tag}.csv", index=False)
    full = manifest["cohort"] == "full"
    plot_report(solver, warm, heur, gains, output, full)
    trace_example(run_directory, solver, output)
    return {"figures": output, "quality": summaries(warm, gains, False)[0],
            "attribution": summaries(warm, gains, False)[1], "references": refs}


def export_combined_figure5(a_directory, b_directory, output):
    """Original A/B cold-certification figure plus the three requested indicators."""
    frames, specs, full_flags = [], [], []
    for directory in (Path(a_directory), Path(b_directory)):
        manifest = json.loads((directory / "manifest.json").read_text())
        p = manifest["config"]["objective"]
        spec = ObjectiveSpec(p["mode"], p["weights"], p["references"])
        legacy_path = Path(__file__).resolve().parent / "historical_reference_values.csv"
        legacy = None
        if spec.historical_compatible:
            if legacy_path.exists():
                legacy = pd.read_csv(legacy_path)
            else:
                from public_sources import historical_references
                legacy = historical_references(spec)
        solver, _, _, _, _ = prepare_study(pd.read_csv(directory / "metrics.csv"), spec, legacy)
        frames.append(solver[solver["Start Method"].isin(["Cold", "ColdReference"])])
        specs.append(spec)
        full_flags.append(manifest["cohort"] == "full")
    if specs[0].fingerprint != specs[1].fingerprint:
        raise ValueError("A/B figures require the same objective and reference scales")
    data = pd.concat(frames, ignore_index=True)
    if set(data["Configuration"]) != {"A", "B"}:
        raise ValueError("Supply one A directory and one B directory")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = data.groupby(["Configuration", "Tasks", "Time Limit sec"], as_index=False).agg(
        n=("Scenario", "size"), certified_pct=("Optimal Certified", lambda s: 100*s.mean()))
    quality = data.groupby(["Configuration", "Time Limit sec"], as_index=False).agg(
        n=("Scenario", "size"), incumbent_pct=("Solver Has Incumbent", lambda s: 100*s.mean()),
        mean_incumbent=("Recorded solver value", "mean"), median_incumbent=("Recorded solver value", "median"),
        known_gap_count=("Solver shortfall pct", "count"), mean_shortfall_pct=("Solver shortfall pct", "mean"))
    summary.to_csv(output / "figure5_certification_by_tasks.csv", index=False)
    quality.to_csv(output / "figure5_budget_quality.csv", index=False)
    fig, axes = plt.subplots(1, 5, figsize=(9, 3.1), sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.76, bottom=.34, wspace=.16)
    colors = {"A": "#0072B2", "B": "#D55E00"}
    budgets = sorted(data["Time Limit sec"].unique())
    for ax, tasks in zip(axes, (20, 30, 40, 50, 60)):
        for c in ("A", "B"):
            rows = summary[summary["Configuration"].eq(c) & summary["Tasks"].eq(tasks)].sort_values("Time Limit sec")
            ax.plot(rows["Time Limit sec"], rows["certified_pct"], color=colors[c],
                    marker="o" if c == "A" else "s", linestyle="-" if c == "A" else "--",
                    label=f"Experiment {c}", markersize=3)
        axes_budget(ax, budgets)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
        ax.set_ylim(-3, 105)
        ax.set_title(f"{tasks} tasks")
        if 1800 in budgets:
            ax.axvspan(600, 1800, color=colors["B"], alpha=.06)
    axes[0].set_ylabel("Certified runs (%)")
    fig.suptitle("Cold-start certification: " + ("full cohorts" if all(full_flags) else "functional pilot; not paper results"),
                 fontsize=10, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .90), ncol=2, frameon=False)
    fig.text(.08, .025, "Independent restarts; long budgets are cold reference jobs. The accompanying CSV reports incumbent\n"
             "availability, value, known-optimum shortfall and their sample counts. Missing budgets are not extrapolated.", fontsize=7)
    save_figure(fig, output, "Figure5_AB_certification")
    return quality
