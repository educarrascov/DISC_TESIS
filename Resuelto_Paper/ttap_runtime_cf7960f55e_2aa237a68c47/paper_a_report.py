"""Publication exports for the five-budget Experiment A protocol.

Raw missing incumbents stay missing. The benchmark separately scores the
available idle plan as zero if cold returned no incumbent. Optimum shortfalls
are paired and never replace unknown optima with best-known incumbents.
"""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from objective_profiles import ObjectiveSpec
from paper_experiment_a import (BENCHMARK_BUDGET, PAPER_BUDGETS, PAPER_TASKS,
                               validate_paper_grid)
from study_analysis import KEYS, prepare_study, summaries

ORDER = ('Cold', 'GreedyBenefit', 'GreedyDensity', 'GBStart', 'GDStart')
NATIVE = ('Cold', 'GBStart', 'GDStart')
LABELS = {'Cold': 'HiGHS cold', 'GreedyBenefit': 'Greedy Benefit',
          'GreedyDensity': 'Greedy Density', 'GBStart': 'HiGHS + Benefit start',
          'GDStart': 'HiGHS + Density start'}
COLORS = {'Cold': '#1f77b4', 'GreedyBenefit': '#ff7f0e', 'GreedyDensity': '#2ca02c',
          'GBStart': '#9467bd', 'GDStart': '#17becf'}
MARKERS = {'Cold': 'o', 'GreedyBenefit': 'o', 'GreedyDensity': 'o',
           'GBStart': 's', 'GDStart': '^'}
STYLES = {'Cold': '-', 'GreedyBenefit': '-', 'GreedyDensity': '-',
          'GBStart': '--', 'GDStart': ':'}


def prepare_paper_data(raw, manifest, allow_pilot=False):
    full = validate_paper_grid(raw, manifest, allow_pilot)
    value = manifest['config']['objective']
    spec = ObjectiveSpec(value['mode'], value['weights'], value['references'])
    solver, warm, heur, gains, refs = prepare_study(raw, spec, require_full=False)
    for column in ('Variables', 'Constraints'):
        values = pd.to_numeric(warm[column], errors='raise')
        if values.isna().any() or values.le(0).any():
            raise ValueError('Invalid MILP size: ' + column)
        if warm.groupby(KEYS)[column].nunique().ne(1).any():
            raise ValueError('Model size changed between treatments: ' + column)
    bench = warm[warm['Time Limit sec'].eq(BENCHMARK_BUDGET)].copy()
    bench['Paper method'] = bench['Start Method']
    completed = pd.to_numeric(bench['Physical Completed Tasks'], errors='raise')
    if completed[bench['Has Incumbent']].isna().any():
        raise ValueError('A returned plan has no completed-task count')
    bench['Available completed tasks'] = completed.where(bench['Has Incumbent'], 0.)
    if (bench['Available completed tasks'].lt(0)
            | bench['Available completed tasks'].gt(bench['Tasks'])).any():
        raise ValueError('Completed-task count is outside the generated task count')
    bench['Available plan'] = bench['Has Incumbent']
    bench['Benchmark shortfall pct'] = bench['Available shortfall pct']
    heur = heur.copy()
    heur['Paper method'] = heur['Start Method'].map(
        {'GBStart': 'GreedyBenefit', 'GDStart': 'GreedyDensity'})
    heur['Available value'] = pd.to_numeric(heur['Normalized Mission Value'], errors='raise')
    heur['Available completed tasks'] = pd.to_numeric(heur['Physical Completed Tasks'], errors='raise')
    if heur[['Available value', 'Available completed tasks']].isna().any().any():
        raise ValueError('A constructive baseline has missing quality metrics')
    heur['Available plan'] = True
    heur['Optimal Certified'] = False
    heur = heur.merge(refs, on=KEYS, how='left', validate='many_to_one')
    if (heur['Available value'] > heur['Certified optimum'] + 1e-6).any():
        raise ValueError('A greedy value exceeds a certified reference')
    heur['Benchmark shortfall pct'] = (100 * (heur['Certified optimum'] - heur['Available value'])
                                        / heur['Certified optimum']).clip(lower=0)
    heur['End-to-end sec'] = pd.to_numeric(heur['Runtime sec'], errors='raise')
    benchmark = pd.concat([bench, heur], ignore_index=True)
    if benchmark.duplicated(KEYS + ['Paper method']).any():
        raise ValueError('Duplicate paper benchmark observations')
    if (benchmark['Available value'].lt(-1e-8) | benchmark['Available value'].gt(1 + 1e-8)).any():
        raise ValueError('Available normalized benefit is outside [0, 1]')
    stats = benchmark.groupby(['Tasks', 'Paper method'], as_index=False).agg(
        n=('Scenario', 'size'), available_plans=('Available plan', 'sum'),
        certified=('Optimal Certified', 'sum'), benefit_mean=('Available value', 'mean'),
        benefit_sd=('Available value', 'std'), tasks_mean=('Available completed tasks', 'mean'),
        tasks_sd=('Available completed tasks', 'std'), n_known_optima=('Certified optimum', 'count'),
        shortfall_mean_pct=('Benchmark shortfall pct', 'mean'),
        runtime_mean_sec=('End-to-end sec', 'mean'), runtime_sd_sec=('End-to-end sec', 'std'),
        runtime_max_sec=('End-to-end sec', 'max'),
        variables_mean=('Variables', 'mean'), constraints_mean=('Constraints', 'mean'))
    return {'full': full, 'spec': spec, 'mip_rel_gap': manifest['config']['mip_rel_gap'],
            'solver': solver, 'warm': warm, 'heuristics': heur,
            'gains': gains, 'references': refs, 'benchmark': benchmark, 'stats': stats}


def _number(value, digits=2):
    return '--' if pd.isna(value) else f'{value:.{digits}f}'


def _mean_sd(mean, sd, digits):
    if pd.isna(sd):
        return '$' + _number(mean, digits) + r'\; (\mathrm{SD}\;\mathrm{n/a})$'
    return '$' + _number(mean, digits) + r' \pm ' + _number(sd, digits) + '$'


def _integer(value):
    return '--' if pd.isna(value) else f'{value:,.0f}'.replace(',', '{,}')


def _table_start(caption, label, headers, tabcolsep):
    return [r'\begin{table}[!htbp]', r'\centering', r'\caption{' + caption + '}',
            r'\label{' + label + '}', r'\scriptsize',
            r'\setlength{\tabcolsep}{' + str(tabcolsep) + 'pt}',
            r'\renewcommand{\arraystretch}{1.08}', r'\begin{tabular}{@{}ccccc@{}}',
            r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']


def latex_tables(data, output):
    stats = data['stats']
    preview = '' if data['full'] else ' Pilot preview; not the 30-replicate study.'
    quality = _table_start('Experiment A: solution quality and certification at the 600-s MILP budget.'
                           + preview, 'tab:expA-quality',
                           ['Tasks', r'\makecell{Normalized available\\benefit\\Mean $\pm$ SD}',
                            r'\makecell{Completed tasks\\Mean $\pm$ SD}',
                            r'\makecell{Certified\\runs}', r'\makecell{Shortfall\\(\%)}'], 2.6)
    runtime = _table_start('Experiment A: computational performance and generated state-arc MILP size.'
                           + preview, 'tab:expA-runtime',
                           ['Tasks', r'\makecell{Runtime\\Mean $\pm$ SD}',
                            r'\makecell{Maximum\\runtime}', r'\makecell{Variables\\mean}',
                            r'\makecell{Constraints\\mean}'], 2.4)
    for index, method in enumerate(ORDER):
        if index:
            quality.append(r'\midrule')
            runtime.append(r'\midrule')
        heading = 'Panel ' + chr(65 + index) + ': ' + LABELS[method]
        unit = 'seconds' if method in NATIVE else 'milliseconds'
        quality.extend([r'\multicolumn{5}{@{}l}{\textit{' + heading + r'}}\\',
                        r'\addlinespace[0.2em]'])
        runtime.extend([r'\multicolumn{5}{@{}l}{\textit{' + heading + ' (' + unit + r')}}\\',
                        r'\addlinespace[0.2em]'])
        for _, row in stats[stats['Paper method'].eq(method)].sort_values('Tasks').iterrows():
            cert = f"${int(row['certified'])}/{int(row['n'])}$" if method in NATIVE else '--'
            quality.append(' & '.join([str(int(row['Tasks'])),
                            _mean_sd(row['benefit_mean'], row['benefit_sd'], 4),
                            _mean_sd(row['tasks_mean'], row['tasks_sd'], 2), cert,
                            _number(row['shortfall_mean_pct'], 1)]) + r' \\')
            factor, digits = (1., 2) if method in NATIVE else (1000., 3)
            runtime.append(' & '.join([str(int(row['Tasks'])),
                            _mean_sd(row['runtime_mean_sec'] * factor,
                                     row['runtime_sd_sec'] * factor, digits),
                            '$' + _number(row['runtime_max_sec'] * factor, digits) + '$',
                            _integer(row['variables_mean']) if method == 'Cold' else '--',
                            _integer(row['constraints_mean']) if method == 'Cold' else '--']) + r' \\')
    counts = stats[stats['Paper method'].eq('Cold')].sort_values('Tasks')
    reference_note = '; '.join(f"{int(r.Tasks)} tasks: {int(r.n_known_optima)}/{int(r.n)}"
                               for r in counts.itertuples())
    ref_note = ('Every generated instance has a certified reference. '
                if counts['n_known_optima'].eq(counts['n']).all()
                else 'Certified-reference counts: ' + reference_note + '. ')
    qnote = ('Means and sample SD summarize the same instances for all five variants. '
             'All MILP variants receive 600 s. Shortfalls are computed per instance against a certified '
             'optimum of the same model and objective, then averaged; unknown optima are excluded. '
             + ref_note + f"Certification denotes HiGHS optimal status at relative MIP tolerance {data['mip_rel_gap']:g}; "
             + 'it is not applicable to the greedy policies.')
    if not data['benchmark']['Available plan'].all():
        qnote += (' If no incumbent is returned, a separately available idle plan contributes zero benefit '
                  'and zero tasks. Raw incumbent values remain missing.')
    if not data['full']:
        qnote += ' SD is unavailable for a one-instance pilot.'
    rnote = ('End-to-end time includes network and matrix generation, native conversion, model loading '
             'and solving. Warm variants additionally include greedy construction, mapping, validation '
             'and submission. Preparation is charged to each treatment although shared within the batch. '
             'Greedy runtimes are in milliseconds. The common MILP size is reported once. '
             'The solver allowance excludes preparation and is not a guaranteed end-to-end deadline.')
    for lines, note, name in [(quality, qnote, 'Table_1_Experiment_A_quality.tex'),
                               (runtime, rnote, 'Table_2_Experiment_A_runtime.tex')]:
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\begin{minipage}{0.98\columnwidth}',
                      r'\footnotesize', note, r'\end{minipage}', r'\end{table}'])
        (output / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _axis(ax):
    ax.grid(True, linestyle='--', linewidth=.6, alpha=.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(.8)


def _save(fig, folder, name, dpi):
    if dpi < 72 or not math.isfinite(float(dpi)):
        raise ValueError('PNG_DPI must be a finite value of at least 72')
    w, h = fig.get_size_inches()
    if w * h * dpi * dpi * 4 > 512 * 1024 ** 2:
        raise ValueError('PNG export would need over 512 MiB for one raster. Use PNG_DPI=600; PDF is vectorial.')
    preview = folder / '_preview'
    preview.mkdir(exist_ok=True)
    fig.savefig(folder / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(folder / (name + '.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(preview / (name + '.png'), dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _pilot_label(fig, full):
    if not full:
        fig.text(.5, .005, 'PILOT PREVIEW - not the 30-replicate study', ha='center', fontsize=7, color='#A02020')


def quality_figure(data, output, dpi):
    bench, stats = data['benchmark'], data['stats']
    native = bench[bench['Paper method'].isin(NATIVE)]
    wide = native.pivot(index=KEYS, columns='Paper method', values='Available value')
    coincident = (native['Optimal Certified'].all() and len(wide) > 0
                  and np.allclose(wide['Cold'], wide['GBStart'], atol=1e-6, rtol=0)
                  and np.allclose(wide['Cold'], wide['GDStart'], atol=1e-6, rtol=0))
    fig, ax = plt.subplots(figsize=(3.5, 2.8), layout='constrained')
    for method in ORDER:
        if coincident and method in ('GBStart', 'GDStart'):
            continue
        rows = stats[stats['Paper method'].eq(method)].sort_values('Tasks')
        label = 'HiGHS variants, certified (600 s)' if coincident and method == 'Cold' else LABELS[method]
        x, y = rows['Tasks'].to_numpy(), rows['benefit_mean'].to_numpy()
        ax.plot(x, y, marker=MARKERS[method], linestyle=STYLES[method], color=COLORS[method],
                markersize=4.5, linewidth=1.7, label=label)
        sd = rows['benefit_sd'].to_numpy()
        if np.isfinite(sd).any():
            ax.fill_between(x, y - sd, y + sd, color=COLORS[method], alpha=.15)
    ax.set_xticks(PAPER_TASKS)
    ax.set_xlim(18, 62)
    ax.set_ylim(0, min(1.02, max(.65, float(stats['benefit_mean'].max()) + .15)))
    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Mean normalized available benefit')
    ax.legend(frameon=True, fontsize=6.4, ncol=1)
    _axis(ax)
    _pilot_label(fig, data['full'])
    _save(fig, output, 'A1_solution_quality_coverage_aware', dpi)
    return bool(coincident)


def effort_figure(data, output, dpi):
    native = data['benchmark'][data['benchmark']['Paper method'].isin(NATIVE)].copy()
    native['Solver Time sec'] = pd.to_numeric(native['Solver Time sec'])
    if native['Solver Time sec'].isna().any() or native['Solver Time sec'].le(0).any():
        raise ValueError('Positive solver times are required for the logarithmic effort figure')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(left=.075, right=.93, bottom=.19, top=.78, wspace=.27)
    cmap, norm = plt.get_cmap('viridis'), plt.Normalize(20, 60)
    rng = np.random.default_rng(7)
    for index, method in enumerate(NATIVE):
        rows = native[native['Paper method'].eq(method)].sort_values(['Tasks', 'Seed'])
        x = rows['Tasks'].to_numpy() + (index - 1) * 1.1 + rng.uniform(-.35, .35, len(rows))
        for certified in (True, False):
            mask = rows['Optimal Certified'].eq(certified).to_numpy()
            colors = cmap(norm(rows.loc[mask, 'Tasks'].to_numpy()))
            for ax, xx in [(axes[0], x[mask]), (axes[1], rows.loc[mask, 'Variables'])]:
                ax.scatter(xx, rows.loc[mask, 'Solver Time sec'], marker=MARKERS[method], s=20,
                           facecolors=colors if certified else 'none',
                           edgecolors='#333333' if certified else colors, linewidths=.45, alpha=.80)
        summary = rows.groupby('Tasks')['Solver Time sec'].agg(
            median='median', q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
        xx = summary.index.to_numpy()
        axes[0].plot(xx, summary['median'], color=COLORS[method], linestyle=STYLES[method],
                     marker=MARKERS[method], markersize=4, linewidth=1.25)
        axes[0].fill_between(xx, summary['q1'], summary['q3'], color=COLORS[method], alpha=.10)
    for ax in axes:
        ax.set_yscale('log')
        ax.axhline(BENCHMARK_BUDGET, color='#D62728', linestyle='--', linewidth=1)
        _axis(ax)
        ax.set_ylabel('HiGHS solver time (s, log scale)')
        ax.set_ylim(max(.001, native['Solver Time sec'].min() * .6),
                    max(600., native['Solver Time sec'].max()) * 1.5)
    axes[0].set_xticks(PAPER_TASKS)
    axes[0].set_xlim(17, 63)
    axes[0].set_xlabel('Number of tasks')
    axes[0].set_title('(a) Computational effort', fontsize=9)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Feasible state-transition arcs\n(binary decision variables)')
    axes[1].set_title('(b) State-arc size and runtime', fontsize=9)
    handles = [Line2D([], [], marker=MARKERS[m], color=COLORS[m], linestyle=STYLES[m],
                      label=LABELS[m]) for m in NATIVE]
    handles += [Line2D([], [], marker='o', color='black', linestyle='', label='Certified (filled)'),
                Line2D([], [], marker='o', markerfacecolor='none', color='black', linestyle='',
                       label='Not certified (open)'),
                Line2D([], [], color='#D62728', linestyle='--', label='600-s solver allowance')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99), ncol=3,
               fontsize=7, frameon=False)
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[1],
                           fraction=.045, pad=.03)
    colorbar.set_label('Number of tasks', fontsize=8)
    _pilot_label(fig, data['full'])
    _save(fig, output, 'A8_computational_effort', dpi)


def budget_figures(data, output, dpi):
    quality, attr = summaries(data['warm'], data['gains'], by_task=True)
    definitions = [('certified_pct', 'Optimality\ncertified (%)', 'A3_certification_by_tasks_and_budget', 100),
                   ('solver_incumbents_pct', 'Solver-returned\nincumbent (%)', 'A9_budget_feasibility', 100),
                   ('mean_solver_incumbent', 'Mean recorded\nsolver benefit', 'A10_budget_incumbent_value', 1),
                   ('mean_solver_shortfall_pct', 'Mean shortfall to\nknown optimum (%)', 'A11_budget_optimality_shortfall', 100)]
    for column, ylabel, name, maximum in definitions:
        fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.65), sharey=True)
        fig.subplots_adjust(left=.08, right=.99, bottom=.25, top=.72, wspace=.15)
        for ax, count in zip(axes, PAPER_TASKS):
            for method in NATIVE:
                rows = quality[quality['Tasks'].eq(count) & quality['Start Method'].eq(method)]
                ax.plot(rows['Time Limit sec'], rows[column], color=COLORS[method],
                        marker=MARKERS[method], linestyle=STYLES[method], markersize=3, linewidth=1.2)
            if not quality['Tasks'].eq(count).any():
                ax.text(.5, .5, 'Not run', ha='center', transform=ax.transAxes, fontsize=7)
            ax.set_xscale('log')
            ax.set_xticks(PAPER_BUDGETS, ['1', '5', '15', '60', '600'], rotation=45, fontsize=6.5)
            ax.set_ylim(-.03 * maximum, 1.05 * maximum)
            ax.set_title(f'{count} tasks', fontsize=8)
            ax.set_xlabel('Solver allowance (s)', fontsize=7)
            _axis(ax)
        axes[0].set_ylabel(ylabel, fontsize=7.5)
        handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle=STYLES[m], label=LABELS[m])
                   for m in NATIVE]
        fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=7.5)
        _pilot_label(fig, data['full'])
        _save(fig, output, name, dpi)
    # The common fallback comparison distinguishes possessing a greedy from injecting it.
    overall = summaries(data['warm'], data['gains'], by_task=False)[1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout='constrained')
    for ax, column, ylabel in zip(axes,
            ('mean_gain_over_greedy', 'mean_injection_gain'),
            ('Gain over initial greedy', 'Gain over cold + same greedy fallback')):
        for method in ('GBStart', 'GDStart'):
            rows = overall[overall['Start Method'].eq(method)].sort_values('Time Limit sec')
            ax.plot(rows['Time Limit sec'], rows[column], color=COLORS[method], marker=MARKERS[method],
                    linestyle=STYLES[method], label=LABELS[method])
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xscale('log')
        ax.set_xticks(PAPER_BUDGETS, ['1', '5', '15', '60', '600'], rotation=45)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_xlabel('Solver allowance (s)')
        ax.legend(fontsize=6.5)
        _axis(ax)
    _pilot_label(fig, data['full'])
    _save(fig, output, 'A12_warm_start_attribution', dpi)
    return quality, attr


def _impact_fragment(spec):
    if spec.mode != 'multidimensional':
        return ('% This run uses the primary-output objective. Do not paste the multidimensional definition.\n')
    w, r = spec.weights, spec.references
    return r'''% Replace the primary-output part of Section 3.3; keep the existing S_i(t).
% Apply this definition consistently in A, B and Talcahuano before reporting them together.
The task quantities represent distinct delivered outputs: medical evacuees,
non-medical passengers, and kilograms of supplied cargo. Personnel counts
exclude patients already counted in the medical component; cargo excludes
their body mass and equipment used only to support the operation.
The potential value of task $i$ is
\[
\phi_i=\lambda_m\frac{m_i}{R_m}
      +\lambda_p\frac{p_i}{R_p}
      +\lambda_q\frac{q_i}{R_q}.
\]
''' + (f'The weights are $(\\lambda_m,\\lambda_p,\\lambda_q)='
        f"({w['medical']:g},{w['personnel']:g},{w['cargo']:g})$, and the references are "
        f"$(R_m,R_p,R_q)=({r['medical']:g},{r['personnel']:g},{r['cargo']:g})$.\n") + r'''
The references are fixed demand-generator scales and remain constant across
instances and fleet configurations. The weights express compensatory
preferences; they do not impose absolute medical precedence and have not
been clinically calibrated. Task classes remain descriptive categories.
With the temporal satisfaction function $S_i$ defined below, let
\[
V=\sum_{j\in I}\phi_j,
\qquad b_a=\frac{\phi_{i(a)}S_{i(a)}(t_a^f)}{V},\quad a\in A_h^T.
\]
Return, recovery and termination arcs have zero benefit. The same coefficients
are used by both constructive policies and by every MILP initialization strategy.
% In Section 4, replace w_i eta_i S_i(t)/V by phi_i S_i(t)/V in both greedy scores.
'''


def section_fragment(data, output, coincident):
    # A final-looking replacement is deliberately not emitted for a pilot.
    if not data['full']:
        (output / 'PILOT_ONLY.txt').write_text(
            'These files are a preview. Section_4_1.tex is generated only after all 150 instances '
            'and 2250 solver jobs are present. Do not replace paper results with this sample.\n')
        return
    stats, benchmark = data['stats'], data['benchmark']
    totals = stats[stats['Paper method'].isin(NATIVE)].groupby('Paper method')['certified'].sum()
    known = len(data['references'])
    gaps = []
    for method in ('GreedyBenefit', 'GreedyDensity'):
        rows = stats[stats['Paper method'].eq(method)]['shortfall_mean_pct'].dropna()
        if not rows.empty:
            if abs(rows.max() - rows.min()) < .05:
                gaps.append(f"For {LABELS[method]}, mean paired shortfalls were {rows.mean():.1f}\\% at every task count.")
            else:
                gaps.append(f"For {LABELS[method]}, mean paired shortfalls ranged from {rows.min():.1f}\\% to {rows.max():.1f}\\%.")
    comparison = (' '.join(gaps) + ' '
                  if gaps else 'No certified reference was available for a shortfall comparison. ')
    paragraph = (f"At 600 s, cold HiGHS certified {int(totals['Cold'])}/150 instances; "
                 f"the Benefit and Density starts certified {int(totals['GBStart'])}/150 and "
                 f"{int(totals['GDStart'])}/150, respectively. " + comparison
                 + f'Shortfall calculations use the {known}/150 instances with a certified reference '
                 'from this objective and model; unknown optima are excluded.\n')
    changes = []
    cold = benchmark[benchmark['Paper method'].eq('Cold')][
        KEYS + ['Optimal Certified', 'End-to-end sec']].rename(
            columns={'Optimal Certified': 'Cold certified', 'End-to-end sec': 'Cold seconds'})
    for method in ('GBStart', 'GDStart'):
        paired = benchmark[benchmark['Paper method'].eq(method)].merge(cold, on=KEYS, validate='one_to_one')
        paired = paired[paired['Optimal Certified'] & paired['Cold certified']]
        if not paired.empty:
            delta = (paired['End-to-end sec'] - paired['Cold seconds']).median()
            changes.append(f"{LABELS[method]}: {delta:+.2f} s over {len(paired)} matched certified pairs")
    timing = ('On pairs where both variants certified, median warm-minus-cold end-to-end changes were '
              + '; '.join(changes) + '. Positive changes mean that the warm variant took longer. '
              if changes else 'No paired certification comparison of end-to-end times was available. ')
    overlap = ('The three MILP curves coincide and are drawn once because all three variants certified '
               'every instance and their paired benefits agree within numerical tolerance. '
               if coincident else 'The MILP initialization strategies are drawn separately. ')
    text = r'''\subsection{Experiment A: Controlled Repeated-Instance Scaling}
\label{subsec:experiment-a}
Experiment A uses four helicopters and four demand areas, with 20, 30, 40,
50 and 60 tasks and 30 generated instances per size. Each instance is solved
by Greedy Benefit and Greedy Density, and by HiGHS without a start, with a
Benefit start, and with a Density start. The three MILP variants share the
same matrix and solver settings. Each receives independent solve allowances
of 1, 5, 15, 60 and 600 s. The 600-s runs provide the main comparison below;
the shorter budgets are examined in Section~4.3.
The multidimensional benefit defined in Section~3.3 is
used throughout.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\columnwidth]{A1_solution_quality_coverage_aware.pdf}
\caption{Normalized available benefit in Experiment A at the 600-s MILP
allowance. Lines denote means over the same 30 instances at each task count;
shaded regions denote one sample standard deviation. An externally available
idle plan is scored as zero when cold returns no incumbent. ''' + overlap + r'''}
\label{fig:expA-quality}
\end{figure}

''' + paragraph + r'''
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{A8_computational_effort.pdf}
\caption{Experiment A computational effort for the three initialization
strategies. Panel (a) shows all solver times, with medians and interquartile
ranges. Panel (b) relates solver time to generated state-transition arcs.
Marker shape identifies the initialization strategy; filled and open markers
denote certified and uncertified runs. The dashed line is the 600-s solver
allowance. Table~\ref{tab:expA-runtime} reports end-to-end time, including
preparation.}
\label{fig:expA-effort}
\end{figure*}

''' + timing + r'''
Warm-start availability by itself does not establish a search improvement:
the constructive plan is already available before optimization. The budget
analysis therefore also compares each warm result with cold retaining the
same greedy plan as an external fallback.

\input{Table_1_Experiment_A_quality.tex}
\input{Table_2_Experiment_A_runtime.tex}
'''
    if data['spec'].mode != 'multidimensional':
        text = text.replace('The multidimensional benefit defined in Section~3.3 is',
                            'The primary-output benefit defined in Section~3.3 is')
    (output / 'Section_4_1.tex').write_text(text, encoding='utf-8')


def export_paper_a(run_directory, png_dpi=600, allow_pilot=False):
    root = Path(run_directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    raw = pd.read_csv(root / 'metrics.csv')
    data = prepare_paper_data(raw, manifest, allow_pilot)
    output = root / ('paper' if data['full'] else 'paper_preview')
    output.mkdir(exist_ok=True)
    data['benchmark'].to_csv(output / 'benchmark_600_paired_rows.csv', index=False)
    data['stats'].to_csv(output / 'Tables_1_2_statistics.csv', index=False)
    data['references'].to_csv(output / 'certified_reference_values.csv', index=False)
    data['gains'].to_csv(output / 'paired_warm_start_gains.csv', index=False)
    latex_tables(data, output)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        coincident = quality_figure(data, output, png_dpi)
        effort_figure(data, output, png_dpi)
        budget_quality, attribution = budget_figures(data, output, png_dpi)
    budget_quality.to_csv(output / 'budget_quality_by_task.csv', index=False)
    attribution.to_csv(output / 'warm_attribution_by_task.csv', index=False)
    section_fragment(data, output, coincident)
    (output / 'Section_3_3_impact_update.tex').write_text(_impact_fragment(data['spec']), encoding='utf-8')
    meta = {'full_cohort': data['full'], 'benchmark_budget_sec': BENCHMARK_BUDGET,
            'solver_budgets_sec': list(PAPER_BUDGETS), 'instances': len(raw[KEYS].drop_duplicates()),
            'solver_runs': len(data['warm']), 'heuristic_runs': len(data['heuristics']),
            'known_certified_optima': len(data['references']), 'coincident_certified_MILP_curves': coincident,
            'objective': data['spec'].as_dict(), 'png_dpi': png_dpi,
            'quality_estimand': 'mean available value on every generated instance; idle=0 if no returned plan',
            'budget_quality_estimand': 'solver-returned incumbents; missing incumbents excluded with counts',
            'gap_definition': 'per-instance shortfall to certified optimum, averaged on known references only'}
    (output / 'export_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    (output / 'OVERLEAF_README.txt').write_text(
        'Preamble packages: booktabs, makecell, graphicx.\n'
        'Upload the two main PDF figures and the .tex files from this folder.\n'
        'Replace Section 4.1 with Section_4_1.tex only after a full run.\n'
        'Keep existing figure labels if your project uses other names.\n'
        'Update Section 3.3 and both greedy-score formulas before mixing these values with old results.\n'
        'Use matching objective settings when rerunning B and Talcahuano.\n'
        'A3 and A9-A12 support Section 4.3; the mean/median/count CSVs preserve the denominators.\n'
        'No CSV, figure or manuscript is uploaded to GitHub by this program.\n')
    return {'directory': output, 'quality': data['stats'], 'budget_quality': budget_quality,
            'attribution': attribution, 'metadata': meta}
