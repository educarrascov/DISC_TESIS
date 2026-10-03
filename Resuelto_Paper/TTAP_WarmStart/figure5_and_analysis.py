"""Rebuild budget results from the original CSVs and optional new warm runs.

Recorded missing incumbents remain missing. A separate available-value score
uses the known idle plan (zero) when no incumbent was recorded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def booleans(series):
    text = series.astype(str).str.lower().str.strip()
    known = text.isin(["true", "false", "1", "0", "nan", "none", ""])
    if not known.all():
        raise ValueError(f"Unexpected boolean values: {text[~known].unique()}")
    return text.isin(["true", "1"])


def prepare(frame, configuration=None):
    frame = frame[frame["Method"].astype(str).str.startswith("HiGHS-")].copy()
    if configuration is not None:
        frame["Configuration"] = configuration
    if "Start Method" not in frame:
        frame["Start Method"] = "Historical cold"
    frame["Budget sec"] = pd.to_numeric(frame["Time Limit sec"], errors="raise")
    frame["Has Incumbent"] = booleans(frame["Has Incumbent"])
    frame["Optimal Certified"] = booleans(frame["Optimal Certified"])
    frame["Solver Has Incumbent"] = booleans(
        frame["Solver Has Incumbent"] if "Solver Has Incumbent" in frame
        else frame["Has Incumbent"])
    frame["Incumbent value"] = pd.to_numeric(frame["Normalized Mission Value"], errors="coerce")
    frame["Solver incumbent value"] = pd.to_numeric(
        frame["Solver Incumbent Value"] if "Solver Incumbent Value" in frame
        else frame["Normalized Mission Value"], errors="coerce")
    if frame.loc[frame["Has Incumbent"], "Incumbent value"].isna().any():
        raise ValueError("Recorded incumbent without a finite objective")
    if frame.loc[~frame["Has Incumbent"], "Incumbent value"].notna().any():
        raise ValueError("Recorded value without an incumbent")
    if ((frame["Incumbent value"] < -1e-8) | (frame["Incumbent value"] > 1 + 1e-8)).any():
        raise ValueError("Normalized objective outside [0,1]")
    frame["Available value"] = frame["Incumbent value"].fillna(0.0)
    frame["Positive plan"] = frame["Has Incumbent"] & frame["Available value"].gt(1e-10)
    keys = ["Configuration", "Scenario", "Budget sec", "Start Method"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate scenario/budget/treatment rows")
    if (frame["Optimal Certified"] & ~frame["Solver Has Incumbent"]).any():
        raise ValueError("Certified run without a solver incumbent")
    return frame


def references(historical):
    rows = []
    for configuration, budget in (("A", 600), ("B", 1800)):
        data = historical[historical["Configuration"].eq(configuration)]
        reference = data[data["Budget sec"].eq(budget) & data["Optimal Certified"] &
                         data["Incumbent value"].gt(0)].copy()
        reference = reference[["Configuration", "Scenario", "Tasks", "Incumbent value"]]
        reference = reference.rename(columns={"Incumbent value": "Certified optimum"})
        reference["Reference budget sec"] = budget
        rows.append(reference)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["Configuration", "Scenario", "Tasks"]).any():
        raise ValueError("Nonunique certified references")
    return result


def attach_references(frame, reference):
    result = frame.merge(reference, on=["Configuration", "Scenario", "Tasks"],
                         how="left", validate="many_to_one")
    if (result["Available value"] > result["Certified optimum"] + 1e-6).any():
        raise ValueError("Value exceeds certified reference: check model/objective/instance identity")
    # Clamp only sub-tolerance numerical differences, not genuinely better values.
    available_gap = 100 * (result["Certified optimum"] - result["Available value"]) / result["Certified optimum"]
    result["Available shortfall pct"] = available_gap.clip(lower=0)
    result["Incumbent shortfall pct"] = result["Available shortfall pct"].where(result["Has Incumbent"])
    return result


def summarize(frame, include_tasks=True):
    keys = ["Configuration", "Start Method"] + (["Tasks"] if include_tasks else []) + ["Budget sec"]
    result = frame.groupby(keys, observed=True).agg(
        instances=("Scenario", "size"),
        incumbents=("Has Incumbent", "sum"),
        solver_incumbents=("Solver Has Incumbent", "sum"),
        positive_plans=("Positive plan", "sum"),
        certified=("Optimal Certified", "sum"),
        mean_incumbent=("Incumbent value", "mean"),
        median_incumbent=("Incumbent value", "median"),
        mean_solver_incumbent=("Solver incumbent value", "mean"),
        q25_incumbent=("Incumbent value", lambda s: s.quantile(.25)),
        q75_incumbent=("Incumbent value", lambda s: s.quantile(.75)),
        mean_available=("Available value", "mean"),
        median_available=("Available value", "median"),
        matched_certified_references=("Certified optimum", "count"),
        matched_incumbents=("Incumbent shortfall pct", "count"),
        mean_incumbent_shortfall_pct=("Incumbent shortfall pct", "mean"),
        median_incumbent_shortfall_pct=("Incumbent shortfall pct", "median"),
        mean_available_shortfall_pct=("Available shortfall pct", "mean"),
        median_available_shortfall_pct=("Available shortfall pct", "median"),
    ).reset_index()
    for label in ("incumbents", "solver_incumbents", "positive_plans", "certified"):
        result[label + "_pct"] = 100 * result[label] / result["instances"]
    return result


def plot_current(summary, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(7.8, 9.2))
    fig.subplots_adjust(left=.07, right=.97, bottom=.17, top=.90, hspace=.62, wspace=.22)
    definitions = [
        ("incumbents_pct", "Feasible incumbent / positive plan (%)", "Blues", 100),
        ("mean_incumbent", "Mean best incumbent (recorded solutions)", "Blues", 1),
        ("mean_available_shortfall_pct", "Mean available shortfall to certified optimum (%)", "Oranges", 100),
        ("certified_pct", "Certified optimality (%)", "Blues", 100),
    ]
    tasks = sorted(summary["Tasks"].unique())
    for col, config in enumerate(["A", "B"]):
        data = summary[summary["Configuration"].eq(config)]
        budgets = sorted(data["Budget sec"].unique())
        lookup = data.set_index(["Tasks", "Budget sec"])
        for row, (metric, title, cmap, maximum) in enumerate(definitions):
            ax = axes[row, col]
            grid = np.array([[lookup.loc[(task, b), metric] for b in budgets] for task in tasks])
            palette = plt.get_cmap(cmap).copy()
            palette.set_bad("#EEEEEE")
            image = ax.imshow(np.ma.masked_invalid(grid), cmap=palette, vmin=0, vmax=maximum,
                              aspect="auto", interpolation="nearest")
            for i, task in enumerate(tasks):
                for j, budget in enumerate(budgets):
                    record = lookup.loc[(task, budget)]
                    value = grid[i, j]
                    if row == 0:
                        label = f"{record['incumbents_pct']:.0f}/{record['positive_plans_pct']:.0f}"
                    elif row == 1:
                        label = f"{value:.3f}" if np.isfinite(value) else "n.a."
                    else:
                        label = f"{value:.1f}" if np.isfinite(value) else "n.a."
                    color = "white" if np.isfinite(value) and value > maximum * .65 else "#222222"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8.0, color=color)
            ax.set_xticks(range(len(budgets)), [f"{b:g}" for b in budgets])
            ax.set_yticks(range(len(tasks)), [str(t) for t in tasks])
            ax.set_xlabel("Solver time budget (s)")
            if col == 0:
                ax.set_ylabel("Tasks")
            ax.set_title(title, fontsize=8.8, pad=7)
            if row == 0:
                position = ax.get_position()
                fig.text(position.x0 + position.width / 2, .952,
                         f"Experiment {config} ({'4 × 4' if config == 'A' else '6 × 7'})",
                         ha="center", fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.text(.5, .99, "Availability, incumbent quality, and optimality certification", ha="center", fontsize=11.5)
    fig.text(.05, .075,
             "30 instances per task count. First row: feasible incumbent / positive plan.\n"
             "Mean incumbent excludes missing solutions (n is reported in the CSV). Shortfall uses a zero-value\n"
             "idle fallback when no incumbent is recorded. Certified-reference samples: A, n=30 at every size;\n"
             "B, n=30, 30, 27, 21, 9 for 20–60 tasks. Budgets are independent restarts, not one continuous run.",
             fontsize=7.4, ha="left", va="center", color="#444444")
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"Figure5_current.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def paired_warm_differences(frame):
    keys = ["Configuration", "Scenario", "Tasks", "Budget sec"]
    cold = frame[frame["Start Method"].eq("Cold")][keys + ["Available value"]].rename(
        columns={"Available value": "Cold available value"})
    warm = frame[frame["Start Method"].isin(["GBStart", "GDStart"])].merge(
        cold, on=keys, how="inner", validate="many_to_one")
    if len(warm) != len(frame[frame["Start Method"].isin(["GBStart", "GDStart"])]):
        raise ValueError("Missing cold control for a warm-start treatment")
    warm["Paired value improvement"] = warm["Available value"] - warm["Cold available value"]
    return warm


def plot_warm(summary, out):
    colors = {"Cold": "#6C757D", "GBStart": "#264653", "GDStart": "#2A9D8F"}
    labels = {"Cold": "HiGHS cold", "GBStart": "HiGHS + Greedy Benefit", "GDStart": "HiGHS + Greedy Density"}
    for config in summary["Configuration"].unique():
        data = summary[summary["Configuration"].eq(config) & summary["Start Method"].isin(colors)]
        sizes = sorted(data["Tasks"].unique())
        fig, axes = plt.subplots(3, len(sizes), figsize=(3.0 * len(sizes), 7.7), squeeze=False,
                                 sharey="row", constrained_layout=True)
        for col, tasks in enumerate(sizes):
            for method in colors:
                rows = data[data["Tasks"].eq(tasks) & data["Start Method"].eq(method)].sort_values("Budget sec")
                for row, metric in enumerate(["incumbents_pct", "mean_available", "mean_available_shortfall_pct"]):
                    axes[row, col].plot(rows["Budget sec"], rows[metric], color=colors[method],
                                        marker="o", markersize=4, linewidth=1.6, label=labels[method])
            for row in range(3):
                ax = axes[row, col]
                budgets = sorted(data["Budget sec"].unique())
                ax.set_xscale("log")
                ax.set_xticks(budgets, [f"{x:g}" for x in budgets])
                ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
                ax.spines[["top", "right"]].set_visible(False)
                if row == 2:
                    ax.set_xlabel("Solver time budget (s)")
            axes[0, col].set_title(f"{tasks} tasks")
        for row, label in enumerate(["Feasible plan available (%)", "Mean available value", "Shortfall to certified optimum (%)"]):
            axes[row, 0].set_ylabel(label)
        axes[0, 0].set_ylim(-3, 103)
        axes[1, 0].set_ylim(0, 1)
        axes[2, 0].set_ylim(-3, 103)
        axes[0, 0].legend(frameon=False, fontsize=7)
        fig.suptitle(f"Experiment {config}: cold and greedy-initialized HiGHS", fontsize=12)
        for extension in ["png", "pdf"]:
            fig.savefig(out / f"Figure_WarmStart_{config}.{extension}", dpi=300,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--warm-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    historical = pd.concat([prepare(pd.read_csv(args.a), "A"), prepare(pd.read_csv(args.b), "B")], ignore_index=True)
    reference = references(historical)
    current = attach_references(historical, reference)
    detailed = summarize(current)
    detailed.to_csv(args.output / "Figure5_current_by_task.csv", index=False)
    summarize(current, include_tasks=False).to_csv(args.output / "Figure5_current_overall.csv", index=False)
    reference.to_csv(args.output / "certified_references.csv", index=False)
    plot_current(detailed, args.output)
    if args.warm_csv:
        warm = prepare(pd.concat([pd.read_csv(p) for p in args.warm_csv], ignore_index=True))
        expected = "primary-output_matched-v3_fixed-references-3-3-500"
        if "Objective Version" not in warm or not warm["Objective Version"].eq(expected).all():
            raise ValueError("Warm runs do not identify the same historical objective")
        warm = attach_references(warm, reference)
        summary = summarize(warm)
        summary.to_csv(args.output / "warm_summary_by_task.csv", index=False)
        warm.to_csv(args.output / "warm_paired_to_certified.csv", index=False)
        paired = paired_warm_differences(warm)
        paired.to_csv(args.output / "warm_vs_cold_paired.csv", index=False)
        paired.groupby(["Configuration", "Tasks", "Start Method", "Budget sec"])["Paired value improvement"].agg(
            ["count", "mean", "median", "std"]).reset_index().to_csv(args.output / "warm_vs_cold_summary.csv", index=False)
        plot_warm(summary, args.output)
    print(f"Analysis saved in {args.output}")


if __name__ == "__main__":
    main()
