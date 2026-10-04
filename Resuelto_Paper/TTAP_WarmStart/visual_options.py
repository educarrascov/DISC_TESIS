"""Five Figure-5 layouts from the unchanged historical TTAP data.

Every layout is generated from the same validated data. Missing incumbents
remain missing in conditional benefit and shortfall statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from figure5_and_analysis import prepare, references, attach_references, summarize

ROOT = Path(__file__).resolve().parent
A_RELATIVE = Path("ttap_experiment_A_4x4_impact_anytime_traced") / "metrics_A4x4_paper_20260808_112401-0400_matched_v3_impact_anytime.csv"
B_RELATIVE = Path("ttap_experiment_B_6x7_stress_impact_anytime_traced") / "metrics_B6x7_stress_paper_20260825_195505-0400_matched_v3_impact_anytime_old6x7_operational_config.csv"
COLORS = {"A": "#0072B2", "B": "#D55E00"}
MARKERS = {"A": "o", "B": "s"}
TASKS = [20, 30, 40, 50, 60]
ALL_BUDGETS = [1, 5, 15, 60, 600, 1800]
LABELS = {"A": "A (4 x 4)", "B": "B (6 x 7)"}
NOTES = ("Separate solver restarts; connecting lines do not represent one search.\n"
         "Incumbent statistics exclude missing solutions. Shortfalls use certified-reference subsets only.\n"
         "Certified references: A=150, B=117; valid shortfall sample sizes vary by budget.")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "lines.linewidth": 1.5,
                         "lines.markersize": 3.7})


def validate_historical(frame):
    for config, budgets in (("A", ALL_BUDGETS[:-1]), ("B", ALL_BUDGETS)):
        data = frame[frame["Configuration"].eq(config)]
        expected = {(task, seed, float(b)) for task in TASKS
                    for seed in range(1, 31) for b in budgets}
        observed = set(map(tuple, data[["Tasks", "Seed", "Budget sec"]].to_numpy()))
        if observed != expected or len(data) != len(expected):
            raise ValueError(f"{config}: use the full original 30-seed historical CSV")
    certified = frame[frame["Optimal Certified"]].groupby(
        ["Configuration", "Budget sec"]).size()
    if certified.loc[("A", 600)] != 150 or certified.loc[("B", 1800)] != 117:
        raise ValueError("Historical references differ from the declared source commit")


def load_data(a, b):
    raw_a, raw_b = pd.read_csv(a), pd.read_csv(b)
    historical = pd.concat([prepare(raw_a, "A"), prepare(raw_b, "B")],
                           ignore_index=True)
    validate_historical(historical)
    reference = references(historical)
    return attach_references(historical, reference), reference


def metric_definitions(stat="mean"):
    return [
        ("incumbents_pct", "Returned incumbent (%)", 100),
        (f"{stat}_incumbent", f"{stat.capitalize()} incumbent benefit", 1),
        (f"{stat}_incumbent_shortfall_pct", f"{stat.capitalize()} incumbent shortfall (%)", 100),
        ("certified_pct", "Optimality certified (%)", 100),
    ]


def budget_axis(ax, maximum=100, xlabel=True, extension=True):
    ax.set_xscale("log")
    ax.set_xlim(.8, 2300)
    ax.set_xticks(ALL_BUDGETS, [str(v) for v in ALL_BUDGETS], rotation=45)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-.025 * maximum, 1.055 * maximum)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    if xlabel:
        ax.set_xlabel("Solver budget (s)")
    if extension:
        ax.axvspan(600, 1800, color=COLORS["B"], alpha=.07, zorder=0)
        ax.axvline(600, color="#888888", linestyle=":", linewidth=.7)


def plot_config(ax, data, metric, config, **kwargs):
    rows = data[data["Configuration"].eq(config)].sort_values("Budget sec")
    ax.plot(rows["Budget sec"], rows[metric], color=COLORS[config],
            marker=MARKERS[config], linestyle="-" if config == "A" else "--",
            markerfacecolor=COLORS[config] if config == "A" else "white",
            label=LABELS[config], **kwargs)


def legend_handles():
    return [Line2D([], [], color=COLORS[c], marker=MARKERS[c],
                   linestyle="-" if c == "A" else "--", label=LABELS[c])
            for c in ("A", "B")]


def save(fig, out, name):
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def option1_frontiers(detail, out, stat="mean"):
    """Keep the original certification strip; put each new metric in its own strip."""
    definitions = metric_definitions(stat)
    ordered = [definitions[3], *definitions[:3]]
    for index, (metric, ylabel, maximum) in enumerate(ordered):
        fig, axes = plt.subplots(1, 5, figsize=(8.5, 2.9), sharey=True)
        fig.subplots_adjust(left=.075, right=.985, top=.79, bottom=.34, wspace=.15)
        for ax, tasks in zip(axes, TASKS):
            data = detail[detail["Tasks"].eq(tasks)]
            for config in ("A", "B"):
                plot_config(ax, data, metric, config)
            budget_axis(ax, maximum)
            ax.set_title(f"{tasks} tasks")
        axes[0].set_ylabel(ylabel)
        fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
                   bbox_to_anchor=(.52, .99), frameon=False)
        fig.text(.075, .02, NOTES + "\nShaded budgets are tested only in B.", fontsize=6.9)
        name = "option1_original_plus_companions" if index == 0 else f"option1_companion_{index}"
        save(fig, out, name)


def option2_summary_curves(overall, out, stat="mean"):
    """Four large panels; pool the five task counts, with 150 runs per budget."""
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.9))
    fig.subplots_adjust(left=.10, right=.97, top=.83, bottom=.27, hspace=.62, wspace=.28)
    for ax, (metric, title, maximum) in zip(axes.flat, metric_definitions(stat)):
        for config in ("A", "B"):
            plot_config(ax, overall, metric, config)
        budget_axis(ax, maximum)
        ax.set_title(title)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2,
               bbox_to_anchor=(.53, .96), frameon=False)
    fig.text(.10, .035, "Pooled task counts: 150 instances per configuration and budget.\n" + NOTES,
             fontsize=7)
    save(fig, out, "option2_four_large_panels")


def option3_task_cards(detail, out, stat="mean"):
    """One card per task count; separate axes preserve the metric units."""
    fig = plt.figure(figsize=(8.3, 9.8))
    outer = fig.add_gridspec(2, 3, left=.075, right=.97, top=.92,
                            bottom=.11, wspace=.33, hspace=.35)
    metrics = metric_definitions(stat)[:3]
    for k, tasks in enumerate(TASKS):
        sub = outer[k // 3, k % 3].subgridspec(3, 1, hspace=.15)
        data = detail[detail["Tasks"].eq(tasks)]
        for row, (metric, title, maximum) in enumerate(metrics):
            ax = fig.add_subplot(sub[row])
            for config in ("A", "B"):
                plot_config(ax, data, metric, config)
                if row == 0:
                    cert = data[data["Configuration"].eq(config)].sort_values("Budget sec")
                    ax.plot(cert["Budget sec"], cert["certified_pct"],
                            color=COLORS[config], linestyle=":", linewidth=1.0)
            budget_axis(ax, maximum, xlabel=(row == 2))
            if row != 2:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_ylabel(["Incumbent (%)", "Benefit", "Shortfall (%)"][row], fontsize=7.5)
            if row == 0:
                ax.set_title(f"{tasks} tasks", fontsize=10)
    key = fig.add_subplot(outer[1, 2])
    key.axis("off")
    key.legend(handles=legend_handles() + [
        Line2D([], [], color="#444444", linestyle="-", label="Returned incumbent"),
        Line2D([], [], color="#444444", linestyle=":", label="Certified optimality")],
        loc="upper left", frameon=False, fontsize=8)
    key.text(.03, .53, "30 instances per task count.\n\n"
             "Benefit: recorded incumbents.\n"
             "Shortfall: matched certified\nreferences only.\n\n"
             "Missing values create gaps;\nthey are not zero benefit.\n\n"
             "Shading: B-only extension.",
             transform=key.transAxes, fontsize=8, va="top", linespacing=1.6)
    fig.text(.075, .020, NOTES, fontsize=7.3)
    save(fig, out, "option3_one_card_per_task_count")


def option4_distributions(frame, out, stat="mean"):
    """Show service/certification categories and all recorded value distributions."""
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 7.9))
    fig.subplots_adjust(left=.105, right=.97, top=.84, bottom=.17,
                        hspace=.65, wspace=.25)
    categories = [
        ("No incumbent", "#E3E6E8"),
        ("Zero-value, uncertified", "#A6AFB7"),
        ("Positive, uncertified", "#E5B260"),
        ("Certified", "#4A8C78"),
    ]
    for col, config in enumerate(("A", "B")):
        data = frame[frame["Configuration"].eq(config)]
        budgets = sorted(data["Budget sec"].unique())
        x = np.arange(len(budgets))
        counts = []
        for budget in budgets:
            group = data[data["Budget sec"].eq(budget)]
            masks = [
                ~group["Has Incumbent"],
                group["Has Incumbent"] & ~group["Optimal Certified"] & ~group["Positive plan"],
                group["Positive plan"] & ~group["Optimal Certified"],
                group["Optimal Certified"],
            ]
            values = np.array([mask.sum() for mask in masks], dtype=float)
            if int(values.sum()) != len(group):
                raise ValueError("Non-exhaustive availability categories")
            counts.append(values / len(group) * 100)
        counts = np.array(counts)
        bottom = np.zeros(len(budgets))
        for i, (_, color) in enumerate(categories):
            axes[0, col].bar(x, counts[:, i], bottom=bottom, color=color, width=.72)
            bottom += counts[:, i]
        axes[0, col].set_title(LABELS[config], fontsize=10)
        axes[0, col].set_ylim(0, 100)
        for row, field in ((1, "Incumbent value"), (2, "Incumbent shortfall pct")):
            ax = axes[row, col]
            for j, budget in enumerate(budgets):
                values = data.loc[data["Budget sec"].eq(budget), field].dropna().to_numpy()
                if len(values):
                    ax.boxplot([values], positions=[j], widths=.5, patch_artist=True,
                               showmeans=True,
                               boxprops={"facecolor": COLORS[config], "alpha": .32},
                               medianprops={"color": "#222222"},
                               meanprops={"marker": "D", "markerfacecolor": "#222222",
                                          "markeredgecolor": "#222222", "markersize": 3},
                               flierprops={"marker": ".", "markersize": 2.5,
                                           "markeredgecolor": COLORS[config]})
                else:
                    ax.text(j, .45, "n.a.", transform=ax.get_xaxis_transform(),
                            ha="center", color="#777777", fontsize=7)
            ax.set_ylim(-.03 if row == 1 else -3, 1.05 if row == 1 else 105)
        for row in range(3):
            ax = axes[row, col]
            ax.set_xticks(x, [str(int(b)) for b in budgets])
            ax.set_xlim(-.6, len(budgets) - .4)
            ax.grid(axis="y", color="#DDDDDD", linewidth=.5)
            ax.set_xlabel("Solver budget (s)")
    for row, label in enumerate(("Run status (%)", "Recorded incumbent benefit", "Incumbent shortfall (%)")):
        axes[row, 0].set_ylabel(label)
    fig.legend(handles=[Patch(facecolor=color, label=name) for name, color in categories],
               loc="upper center", bbox_to_anchor=(.53, .98), ncol=2, frameon=False)
    fig.text(.105, .05, "Pooled task counts. Boxes: interquartile range; line: median; diamond: mean.\n"
             "Whiskers: 1.5 IQR; dots: observations beyond the whiskers, retained in the analysis.\n"
             "Missing incumbents are excluded from boxes; shortfalls require a certified reference.",
             fontsize=6.8)
    save(fig, out, "option4_status_and_distributions")


def option5_original_and_table(detail, overall, out, stat="mean"):
    """Keep original task-wise certification curves and add a one-number-per-cell table."""
    fig = plt.figure(figsize=(8.5, 6.2))
    axes = fig.subplots(1, 5, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.60, top=.84, wspace=.15)
    for ax, tasks in zip(axes, TASKS):
        for config in ("A", "B"):
            plot_config(ax, detail[detail["Tasks"].eq(tasks)], "certified_pct", config)
        budget_axis(ax, 100)
        ax.set_title(f"{tasks} tasks")
    axes[0].set_ylabel("Certified (%)")
    fig.legend(handles=legend_handles(), loc="upper center",
               bbox_to_anchor=(.53, .97), ncol=2, frameon=False)
    table_ax = fig.add_axes([.075, .14, .91, .33])
    table_ax.axis("off")
    columns = ["Config.", "Pooled metric", *[str(b) for b in ALL_BUDGETS]]
    contents, export = [], []
    names = ["Returned incumbent (%)", f"{stat.capitalize()} incumbent benefit",
             f"{stat.capitalize()} incumbent shortfall (%)"]
    for config in ("A", "B"):
        data = overall[overall["Configuration"].eq(config)].set_index("Budget sec")
        for (metric, _, _), label in zip(metric_definitions(stat)[:3], names):
            values = []
            for budget in ALL_BUDGETS:
                value = data.loc[budget, metric] if budget in data.index else np.nan
                values.append(f"{value:.4f}" if metric.endswith("_incumbent") and np.isfinite(value)
                              else f"{value:.1f}" if np.isfinite(value) else "--")
            contents.append([config, label, *values])
            export.append({"Configuration": config, "Metric": label,
                           **dict(zip(map(str, ALL_BUDGETS), values))})
    table = table_ax.table(cellText=contents, colLabels=columns,
                          colWidths=[.065, .30, *([.635 / 6] * 6)], loc="center",
                          cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D2D2D2")
        cell.set_linewidth(.35)
        if row == 0:
            cell.set_facecolor("#ECEFF1")
            cell.set_text_props(weight="bold")
        elif row <= 3:
            cell.set_facecolor("#F4F9FC")
        else:
            cell.set_facecolor("#FFF8F2")
        if col < 2:
            cell.get_text().set_ha("left")
    fig.text(.075, .49, "Table: pooled task counts (150 instances per configuration and budget).",
             fontsize=8)
    fig.text(.075, .035, NOTES + "\n-- indicates unavailable statistics or an untested budget.",
             fontsize=6.9)
    pd.DataFrame(export).to_csv(out / "option5_display_table.csv", index=False)
    def tex_cell(value):
        return str(value).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
    latex_rows = [" & ".join(map(tex_cell, row)) + r" \\" for row in contents]
    latex = "\n".join([r"\begin{tabular}{llrrrrrr}", r"\toprule",
                       " & ".join(map(tex_cell, columns)) + r" \\",
                       r"\midrule", *latex_rows, r"\bottomrule", r"\end{tabular}"])
    (out / "option5_table.tex").write_text(latex, encoding="utf-8")
    save(fig, out, "option5_original_and_numeric_table")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=ROOT.parent / A_RELATIVE)
    parser.add_argument("--b", type=Path, default=ROOT.parent / B_RELATIVE)
    parser.add_argument("--option", choices=["all", "1", "2", "3", "4", "5"], default="all")
    parser.add_argument("--stat", choices=["mean", "median"], default="mean")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "options")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    frame, reference = load_data(args.a, args.b)
    detail = summarize(frame)
    overall = summarize(frame, include_tasks=False)
    detail.to_csv(args.output / "all_options_by_task.csv", index=False)
    overall.to_csv(args.output / "all_options_pooled.csv", index=False)
    reference.to_csv(args.output / "certified_references.csv", index=False)
    frame.to_csv(args.output / "all_options_paired_rows.csv", index=False)
    functions = {
        "1": lambda: option1_frontiers(detail, args.output, args.stat),
        "2": lambda: option2_summary_curves(overall, args.output, args.stat),
        "3": lambda: option3_task_cards(detail, args.output, args.stat),
        "4": lambda: option4_distributions(frame, args.output, args.stat),
        "5": lambda: option5_original_and_table(detail, overall, args.output, args.stat),
    }
    for option in (functions if args.option == "all" else [args.option]):
        functions[option]()
    print(f"Five-layout analysis: {args.output}")


if __name__ == "__main__":
    main()
