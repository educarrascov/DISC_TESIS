"""Separate the greedy plan, the benefit of optimization, and the cost of setup.

Historical CSVs provide certified references only. All solver-performance
comparisons use the NEW matched cold and warm runs.
"""

from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from figure5_and_analysis import prepare, attach_references
from visual_options import ROOT, A_RELATIVE, B_RELATIVE, load_data, style, save

METHODS = ["Cold", "GBStart", "GDStart"]
COLORS = {"Cold": "#555555", "GBStart": "#D55E00", "GDStart": "#009E73",
          "GB": "#CC79A7", "GD": "#0072B2"}
LABELS = {"Cold": "HiGHS without start", "GBStart": "HiGHS + GB",
          "GDStart": "HiGHS + GD", "GB": "Greedy Benefit", "GD": "Greedy Density"}
KEYS = ["Configuration", "Scenario", "Tasks", "Seed"]
PAIR_KEYS = KEYS + ["Budget sec"]
EXPECTED_OBJECTIVE = "primary-output_matched-v3_fixed-references-3-3-500"
EXPECTED_SOURCE = "07dd90042822e2cab84209fe81c9f31ecfe230a8"


def prepare_warm(raw, reference, allow_partial=False):
    new = prepare(raw)
    new = new[new["Start Method"].isin(METHODS)].copy()
    if new.empty:
        raise ValueError("No cold/GBStart/GDStart rows")
    if not new["Objective Version"].eq(EXPECTED_OBJECTIVE).all():
        raise ValueError("Changed objective: historical optima are not valid references")
    if not new["Source Commit"].eq(EXPECTED_SOURCE).all():
        raise ValueError("Changed model source: revalidate historical references before comparison")
    for key, group in new.groupby(PAIR_KEYS, observed=True):
        if set(group["Start Method"]) != set(METHODS):
            raise ValueError(f"Missing matched treatment for {key}")
        for column in ("Model SHA256", "Scenario SHA256", "HiGHS Version",
                       "Threads", "Random Seed"):
            if column not in group or group[column].isna().any() or group[column].nunique() != 1:
                raise ValueError(f"Unmatched {column} for {key}")
    for column in ("HiGHS Version", "Threads", "Random Seed"):
        if new[column].nunique() != 1:
            raise ValueError(f"Do not pool runs with different {column}")
    if not allow_partial:
        expected = {(t, s, b, m) for t in [20, 30, 40, 50, 60]
                    for s in range(1, 31) for b in [1., 5., 15., 60.] for m in METHODS}
        for config, group in new.groupby("Configuration", observed=True):
            observed = set(map(tuple, group[["Tasks", "Seed", "Budget sec", "Start Method"]].to_numpy()))
            if observed != expected:
                raise ValueError(f"{config}: incomplete paper cohort; use --allow-partial for a labelled pilot")
    initial = raw[raw["Method"].isin(["Greedy-Benefit", "Greedy-Density-Operational"])].copy()
    initial["Policy"] = initial["Method"].map(
        {"Greedy-Benefit": "GB", "Greedy-Density-Operational": "GD"})
    if initial.duplicated(KEYS + ["Policy"]).any():
        raise ValueError("Duplicate heuristic measurements: do not combine overlapping execution files")
    greedy = initial.pivot(index=KEYS, columns="Policy",
                           values="Normalized Mission Value").reset_index()
    if not {"GB", "GD"}.issubset(greedy.columns) or greedy[["GB", "GD"]].isna().any().any():
        raise ValueError("Both directly measured greedy baselines are required")
    new = attach_references(new, reference)
    new = new.merge(greedy, on=KEYS, how="left", validate="many_to_one")
    if new[["GB", "GD"]].isna().any().any():
        raise ValueError("Solver instance without matched greedy baselines")
    warm = new[new["Start Method"].isin(["GBStart", "GDStart"])].copy()
    expected_start = np.where(warm["Start Method"].eq("GBStart"), warm["GB"], warm["GD"])
    if not np.allclose(warm["Warm Start Value"], expected_start, atol=1e-8, rtol=0):
        raise ValueError("Submitted start differs from the recorded greedy policy")
    if (warm["Available value"].to_numpy() < expected_start - 1e-8).any():
        raise ValueError("Available plan is worse than its validated initial plan")
    cold = new[new["Start Method"].eq("Cold")][PAIR_KEYS + ["Available value"]].rename(
        columns={"Available value": "Cold value"})
    gain = warm.merge(cold, on=PAIR_KEYS, validate="many_to_one")
    gain["Received greedy value"] = np.where(
        gain["Start Method"].eq("GBStart"), gain["GB"], gain["GD"])
    gain["Cold with same greedy fallback"] = np.maximum(
        gain["Cold value"], gain["Received greedy value"])
    gain["Gain over received greedy"] = gain["Available value"] - gain["Received greedy value"]
    gain["Injection gain over external fallback"] = (
        gain["Available value"] - gain["Cold with same greedy fallback"])
    gain["Improved received greedy"] = gain["Gain over received greedy"].gt(1e-6)
    gain["Beat cold plus same greedy"] = gain["Injection gain over external fallback"].gt(1e-6)
    # Negative injection gains are real and are deliberately retained.
    return new, initial, gain


def pilot_title(partial):
    return "Functional pilot - not paper results" if partial else "Matched cold and warm runs"


def pooled(new):
    return new.groupby(["Configuration", "Start Method", "Budget sec"], observed=True).agg(
        n=("Scenario", "size"),
        solver_incumbent_rate=("Solver Has Incumbent", lambda s: 100 * s.mean()),
        positive_plan_rate=("Positive plan", lambda s: 100 * s.mean()),
        mean_available=("Available value", "mean"),
        mean_recorded_incumbent=("Incumbent value", "mean"),
        mean_shortfall=("Available shortfall pct", "mean"),
        n_references=("Certified optimum", "count"),
        certified_rate=("Optimal Certified", lambda s: 100 * s.mean()),
        mean_runtime=("End-to-end sec", "mean"),
    ).reset_index()


def log_budget_axis(ax, budgets):
    ax.set_xscale("log")
    ax.set_xticks(budgets, [f"{v:g}" for v in budgets])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#DDDDDD", linewidth=.55)


def warm1_policy_and_optimization(new, initial, out, partial):
    summary = pooled(new)
    configs = sorted(new["Configuration"].unique())
    fig, axes = plt.subplots(3, len(configs), figsize=(3.9 * len(configs), 7.7),
                             squeeze=False, sharey="row")
    fig.subplots_adjust(left=.10, right=.97, top=.80, bottom=.17, hspace=.38, wspace=.23)
    for col, config in enumerate(configs):
        data = summary[summary["Configuration"].eq(config)]
        budgets = sorted(data["Budget sec"].unique())
        cases = new[new["Configuration"].eq(config)][KEYS].drop_duplicates()
        axes[0, col].set_title(f"Experiment {config} (n={len(cases)})")
        for method in METHODS:
            rows = data[data["Start Method"].eq(method)].sort_values("Budget sec")
            for row, field in enumerate(["solver_incumbent_rate", "mean_available", "mean_shortfall"]):
                axes[row, col].plot(rows["Budget sec"], rows[field], marker="o",
                                    color=COLORS[method], label=LABELS[method])
        for policy in ("GB", "GD"):
            # Restrict every baseline to the exact measured solver cohort.
            subset = initial[initial["Configuration"].eq(config) & initial["Policy"].eq(policy)].merge(
                cases, on=KEYS, validate="one_to_one")
            value = subset["Normalized Mission Value"].mean()
            axes[1, col].axhline(value, color=COLORS[policy], linestyle="--",
                                linewidth=1.35, label=LABELS[policy])
            refs = new[new["Configuration"].eq(config)][KEYS + ["Certified optimum"]].drop_duplicates(KEYS)
            paired = subset.merge(refs, on=KEYS, validate="one_to_one").dropna(subset=["Certified optimum"])
            shortfall = (100 * (paired["Certified optimum"] - paired["Normalized Mission Value"])
                         / paired["Certified optimum"]).mean()
            if np.isfinite(shortfall):
                axes[2, col].axhline(shortfall, color=COLORS[policy], linestyle="--", linewidth=1.35)
        for row in range(3):
            log_budget_axis(axes[row, col], budgets)
            if row == 2:
                axes[row, col].set_xlabel("Solver budget (s)")
    for row, label in enumerate(["Solver-returned incumbent (%)", "Mean available benefit",
                                 "Available-plan shortfall (%)"]):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_ylim((-.03, 1.03) if row == 1 else (-3, 103))
    fig.suptitle(pilot_title(partial), fontsize=11, y=.985)
    handles = [Line2D([], [], color=COLORS[m], linestyle="-" if m in METHODS else "--",
                      label=LABELS[m]) for m in [*METHODS, "GB", "GD"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .94),
               ncol=3, frameon=False, fontsize=7.2)
    fig.text(.10, .065, "Dashed baselines: directly measured greedy plans. Missing cold incumbents use idle=0\n"
             "only for available value/shortfall. Shortfalls use certified references. Separate restarts;\n"
             "solver budget excludes setup. Initial greedy availability is not a solver improvement.", fontsize=6.8)
    save(fig, out, "warm1_greedy_and_higher_budget_plans")


def warm2_separate_gains(gain, out, partial):
    configs = sorted(gain["Configuration"].unique())
    fig, axes = plt.subplots(2, len(configs), figsize=(3.9 * len(configs), 5.7),
                             squeeze=False)
    fig.subplots_adjust(left=.12, right=.97, top=.80, bottom=.23, hspace=.53, wspace=.3)
    fields = ["Gain over received greedy", "Injection gain over external fallback"]
    for col, config in enumerate(configs):
        data = gain[gain["Configuration"].eq(config)]
        budgets = sorted(data["Budget sec"].unique())
        n = len(data[KEYS].drop_duplicates())
        axes[0, col].set_title(f"Experiment {config} (n={n})")
        for method in ("GBStart", "GDStart"):
            for row, field in enumerate(fields):
                rows = data[data["Start Method"].eq(method)].groupby("Budget sec")[field].agg(
                    mean="mean", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75),
                    n="count").reset_index().sort_values("Budget sec")
                axes[row, col].plot(rows["Budget sec"], rows["mean"], marker="o",
                                    color=COLORS[method], label=LABELS[method])
                if rows["n"].ge(2).all():
                    axes[row, col].fill_between(rows["Budget sec"], rows["q25"], rows["q75"],
                                                color=COLORS[method], alpha=.12)
        for row in range(2):
            ax = axes[row, col]
            ax.axhline(0, color="#777777", linewidth=.8, linestyle=":")
            log_budget_axis(ax, budgets)
            ax.set_xlabel("Solver budget (s)")
            # Preserve negative injection gains and a readable axis for flat pilots.
            if np.ptp(ax.get_ylim()) < .005:
                ax.set_ylim(-.005, .005)
    axes[0, 0].set_ylabel("Benefit added to received greedy")
    axes[1, 0].set_ylabel("Gain from injecting the plan")
    fig.suptitle(pilot_title(partial), fontsize=11, y=.985)
    fig.legend(handles=[Line2D([], [], color=COLORS[m], marker="o", label=LABELS[m])
                        for m in ("GBStart", "GDStart")],
               loc="upper center", bbox_to_anchor=(.55, .94), ncol=2, frameon=False)
    fig.text(.12, .075,
             "Top: warm plan minus its supplied greedy. Bottom: warm plan minus max(new cold, same greedy).\n"
             "Zero means no gain on the stated comparison; negative injection gains are retained.\n"
             "Bands, when present, show the paired-observation IQR, not a confidence interval.", fontsize=6.8)
    save(fig, out, "warm2_credit_to_greedy_vs_injection")


def warm3_setup_and_quality(new, initial, out, partial):
    data = new.copy()
    data["Model preparation"] = data[["Graph Time sec", "Matrix Time sec",
                                     "Native Model Conversion sec", "Model Load sec"]].sum(axis=1)
    data["Start handling"] = data[["Heuristic Time sec", "Warm Start Mapping sec",
                                  "Start Submission sec"]].sum(axis=1)
    if "Start Validation sec" in data:
        data["Start handling"] += data["Start Validation sec"].fillna(0.)
    data["Solve"] = data["Solver Time sec"]
    measured = data[["Model preparation", "Start handling", "Solve"]].sum(axis=1)
    if not np.allclose(measured, data["End-to-end sec"], atol=1e-7, rtol=1e-7):
        raise ValueError("Timing components do not reconstruct the reported preparation+solve time")
    configs = sorted(data["Configuration"].unique())
    fig, axes = plt.subplots(2, len(configs), figsize=(3.9 * len(configs), 7.2), squeeze=False)
    fig.subplots_adjust(left=.11, right=.97, top=.74, bottom=.22, hspace=.62, wspace=.28)
    phase_colors = {"Model preparation": "#6D90A8", "Start handling": "#DEA15B", "Solve": "#A8AFB5"}
    budgets = sorted(data["Budget sec"].unique())
    budget_markers = dict(zip(budgets, cycle(["o", "s", "^", "D", "v", "P", "X"])))
    for col, config in enumerate(configs):
        group = data[data["Configuration"].eq(config)]
        summary = group.groupby(["Budget sec", "Start Method"])[
            ["Model preparation", "Start handling", "Solve", "End-to-end sec", "Available value"]].mean()
        positions, ticks, rows = [], [], []
        for j, budget in enumerate(sorted(group["Budget sec"].unique())):
            for k, method in enumerate(METHODS):
                positions.append(j * 4 + k)
                ticks.append(f"{budget:g}s\n{method.replace('Start', '')}")
                rows.append(summary.loc[(budget, method)])
        bottom = np.zeros(len(rows))
        for field, color in phase_colors.items():
            values = np.array([row[field] for row in rows])
            axes[0, col].bar(positions, values, bottom=bottom, width=.8, color=color)
            bottom += values
        axes[0, col].set_xticks(positions, ticks, fontsize=6.4)
        axes[0, col].set_title(f"Experiment {config}")
        for method in METHODS:
            values = summary.xs(method, level="Start Method").reset_index()
            for _, point in values.iterrows():
                axes[1, col].scatter(point["End-to-end sec"], point["Available value"],
                                     color=COLORS[method], s=37,
                                     marker=budget_markers[point["Budget sec"]],
                                     facecolors="none" if method == "Cold" else COLORS[method],
                                     linewidths=1.1, zorder=3)
        for policy in ("GB", "GD"):
            values = initial[initial["Configuration"].eq(config) & initial["Policy"].eq(policy)]
            axes[1, col].scatter(values["End-to-end sec"].mean(),
                                values["Normalized Mission Value"].mean(),
                                color=COLORS[policy], marker="*", s=65, zorder=4)
        axes[1, col].set_xscale("log")
        axes[1, col].xaxis.set_minor_locator(NullLocator())
        axes[1, col].set_ylim(-.03, 1.05)
        axes[1, col].set_xlabel("Measured preparation + solve (s)")
        for row in range(2):
            axes[row, col].grid(axis="y", color="#DDDDDD", linewidth=.5)
    axes[0, 0].set_ylabel("Mean preparation + solve (s)")
    axes[1, 0].set_ylabel("Mean available benefit")
    fig.suptitle(pilot_title(partial), fontsize=11, y=.985)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=p)
                        for p, c in phase_colors.items()],
               loc="upper center", bbox_to_anchor=(.55, .94), ncol=3, frameon=False, fontsize=7)
    fig.legend(handles=[Line2D([], [], color=COLORS[m], marker="*" if m in ("GB", "GD") else "o",
                               linestyle="none", label=LABELS[m])
                        for m in ["Cold", "GBStart", "GDStart", "GB", "GD"]],
               loc="upper center", bbox_to_anchor=(.55, .88), ncol=3, frameon=False, fontsize=6.8)
    fig.legend(handles=[Line2D([], [], color="#333333", marker=marker, linestyle="none",
                               label=f"{budget:g}s allowance")
                        for budget, marker in budget_markers.items()],
               loc="upper center", bbox_to_anchor=(.55, .79), ncol=4, frameon=False, fontsize=6.5)
    fig.text(.11, .065, "Bars charge setup to each treatment. Scatter shapes denote solver allowances, not elapsed time.\n"
             "Stars at the left are GB (purple) and GD (blue), using their measured construction times.\n"
             "Post-solve decoding/reporting and fingerprint work are excluded; these are not hard deadlines.",
             fontsize=6.8)
    save(fig, out, "warm3_setup_cost_and_quality")


def plot_true_trace(trace, new, out, scenario, budget, partial):
    available = trace[trace["Budget sec"].eq(budget)]
    scenarios = sorted(available["Scenario"].unique())
    if scenario is None:
        if len(scenarios) != 1:
            raise ValueError("Choose --trace-scenario explicitly; never combine different trajectories")
        scenario = scenarios[0]
    available = available[available["Scenario"].eq(scenario)]
    if available.empty or set(available["Start Method"]) != set(METHODS):
        raise ValueError("Trace requires all three methods on one scenario and budget")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    fig.subplots_adjust(left=.10, right=.97, top=.78, bottom=.25)
    for method in METHODS:
        rows = available[available["Start Method"].eq(method)].sort_values(
            "Elapsed solve wall sec", kind="stable")
        ax.step(rows["Elapsed solve wall sec"], rows["Available Value"], where="post",
                color=COLORS[method], label=LABELS[method], linewidth=1.7)
        ax.scatter(rows.iloc[0]["Elapsed solve wall sec"], rows.iloc[0]["Available Value"],
                   color=COLORS[method], marker="s", s=22)
    reference = new.loc[new["Scenario"].eq(scenario), "Certified optimum"].dropna().unique()
    if len(reference) == 1:
        ax.axhline(reference[0], color="#777777", linestyle=":", label="Certified reference")
    ax.set_xlabel("Elapsed solve wall time (s)")
    ax.set_ylabel("Best available benefit")
    ax.set_ylim(-.03, 1.03)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.5)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.30), ncol=2, frameon=False, fontsize=7.4)
    fig.suptitle(f"{pilot_title(partial)}: {scenario}, {budget:g}s allowance", fontsize=10.5, y=.99)
    fig.text(.10, .045, "True callback history within each run. Squares at t=0 denote the validated start or idle fallback;\n"
             "they do not mean HiGHS discovered that plan at t=0. Clock starts after model/start preparation.",
             fontsize=7)
    save(fig, out, "warm4_true_incumbent_trace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=ROOT.parent / A_RELATIVE)
    parser.add_argument("--b", type=Path, default=ROOT.parent / B_RELATIVE)
    parser.add_argument("--warm-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--trace-csv", type=Path)
    parser.add_argument("--trace-scenario")
    parser.add_argument("--trace-budget", type=float, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "warm_options")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    _, reference = load_data(args.a, args.b)
    raw = pd.concat([pd.read_csv(path) for path in args.warm_csv], ignore_index=True)
    new, initial, gain = prepare_warm(raw, reference, args.allow_partial)
    new.to_csv(args.output / "warm_paired_rows.csv", index=False)
    gain.to_csv(args.output / "warm_incremental_gains.csv", index=False)
    pooled(new).to_csv(args.output / "warm_pooled_metrics.csv", index=False)
    gain.groupby(["Configuration", "Tasks", "Start Method", "Budget sec"], observed=True).agg(
        n=("Scenario", "size"), mean_gain_over_greedy=("Gain over received greedy", "mean"),
        mean_injection_gain=("Injection gain over external fallback", "mean"),
        improved_greedy_pct=("Improved received greedy", lambda s: s.mean() * 100),
        beat_external_fallback_pct=("Beat cold plus same greedy", lambda s: s.mean() * 100),
    ).reset_index().to_csv(args.output / "warm_incremental_summary.csv", index=False)
    warm1_policy_and_optimization(new, initial, args.output, args.allow_partial)
    warm2_separate_gains(gain, args.output, args.allow_partial)
    warm3_setup_and_quality(new, initial, args.output, args.allow_partial)
    if args.trace_csv:
        plot_true_trace(pd.read_csv(args.trace_csv), new, args.output, args.trace_scenario,
                        args.trace_budget, args.allow_partial)
    print(f"Warm-start comparisons: {args.output}")


if __name__ == "__main__":
    main()
