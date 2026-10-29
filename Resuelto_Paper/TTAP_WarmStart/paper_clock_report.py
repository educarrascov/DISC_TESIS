"""Observed anytime quality from one real long-budget search per treatment.

Callbacks recorded after a time point cannot contribute to its quality. The
initial validated greedy is credited separately at solve time zero. Preparation
time is excluded; these are observed solve-clock histories, not dispatch deadlines.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_a_report import (COLORS, LABELS, MARKERS, NATIVE, PUBLICATION_STYLE, STYLES,
                            PAPER_DPI, FIGURE_SIZES, _axis, _benchmark, _budgets,
                            _budget_axis, _pilot_label, _save,
                            prepare_paper_data)
from study_analysis import KEYS


def reconstruct_clock_quality(data, traces):
    benchmark = data['benchmark'][data['benchmark']['Paper method'].isin(NATIVE)]
    if traces.empty:
        raise ValueError('No callback history is available')
    if not traces['Objective SHA256'].eq(data['spec'].fingerprint).all():
        raise ValueError('Callback history has another objective')
    traces = traces[traces['Study Stage'].eq('Warm') & traces['Budget sec'].eq(_benchmark(data))].copy()
    grouping = ['Scenario', 'Start Method']
    actual = set(traces[grouping].itertuples(index=False, name=None))
    expected = set(benchmark[['Scenario', 'Paper method']].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError('Missing or extra long-budget callback histories')
    groups = {key: value for key, value in traces.groupby(grouping)}
    rows = []
    for _, run in benchmark.iterrows():
        method, scenario = run['Paper method'], run['Scenario']
        events = groups[(scenario, method)].sort_values('Elapsed solve wall sec')
        elapsed = pd.to_numeric(events['Elapsed solve wall sec'], errors='raise')
        if elapsed.isna().any() or elapsed.lt(0).any():
            raise ValueError('Invalid callback times')
        first = 'Idle fallback available' if method == 'Cold' else 'Validated start available'
        if (events['Event'].eq(first).sum() != 1 or events['Event'].eq('Solve finished').sum() != 1
                or not events.iloc[0]['Event'] == first or elapsed.iloc[0] != 0):
            raise ValueError('Incomplete initial/final history for ' + scenario + '/' + method)
        finish = events[events['Event'].eq('Solve finished')].iloc[0]
        if not np.isclose(finish['Elapsed solve wall sec'], run['Solver Time sec'], atol=1e-6, rtol=0):
            raise ValueError('Callback clock differs from the benchmark run')
        if (events['Tasks'].ne(run['Tasks']).any() or events['Seed'].ne(run['Seed']).any()
                or not np.isclose(finish['Available Value'], run['Available value'], atol=1e-6, rtol=0)):
            raise ValueError('Callback history differs from the paired benchmark')
        initial = 0. if method == 'Cold' else float(run['Warm Start Value'])
        for point in _budgets(data):
            seen = events[elapsed.le(point)]
            candidates = pd.to_numeric(seen['Solver Incumbent Value'], errors='coerce').dropna()
            observed = candidates.max() if not candidates.empty else np.nan
            if np.isfinite(observed) and (observed < -1e-6 or observed > 1+1e-6):
                raise ValueError('Invalid callback objective')
            available = max(initial, observed if np.isfinite(observed) else 0.)
            reference = run['Certified optimum']
            gap = 100 * max(0., reference - available) / reference if pd.notna(reference) else np.nan
            solver_gap = 100 * max(0., reference-observed) / reference if (
                pd.notna(reference) and np.isfinite(observed)) else np.nan
            rows.append({**{key:run[key] for key in KEYS}, 'Start Method':method,
                         'Elapsed solve wall sec':point, 'Source run allowance sec':_benchmark(data),
                         'Observed solver incumbent':observed,
                         'Observed solver plan':np.isfinite(observed),
                         'Available plan':method != 'Cold' or np.isfinite(observed),
                         'Available value':available, 'Positive available plan':available > 1e-10,
                         'Certified by time':bool(run['Optimal Certified'] and run['Solver Time sec'] <= point),
                         'Known optimum':reference, 'Available shortfall pct':gap,
                         'Observed solver shortfall pct':solver_gap})
    return pd.DataFrame(rows)


def clock_summaries(detail, by_task=False):
    keys = (['Tasks'] if by_task else []) + ['Start Method', 'Elapsed solve wall sec']
    result = detail.groupby(keys, as_index=False).agg(
        n=('Scenario', 'size'), available_plans=('Available plan', 'sum'),
        observed_solver_plans=('Observed solver plan', 'sum'),
        positive_available_plans=('Positive available plan', 'sum'),
        mean_available_value=('Available value', 'mean'),
        median_available_value=('Available value', 'median'),
        mean_observed_solver_incumbent=('Observed solver incumbent', 'mean'),
        median_observed_solver_incumbent=('Observed solver incumbent', 'median'),
        known_references=('Known optimum', 'count'),
        mean_available_shortfall_pct=('Available shortfall pct', 'mean'),
        median_available_shortfall_pct=('Available shortfall pct', 'median'),
        observed_solver_gap_count=('Observed solver shortfall pct', 'count'),
        mean_observed_solver_shortfall_pct=('Observed solver shortfall pct', 'mean'),
        certified=('Certified by time', 'sum'))
    result['available_plans_pct'] = 100 * result['available_plans'] / result['n']
    result['observed_solver_plans_pct'] = 100 * result['observed_solver_plans'] / result['n']
    return result


def export_clock_quality(run_directory, experiment='A', png_dpi=PAPER_DPI, allow_pilot=False, data=None):
    root = Path(run_directory)
    if data is None:
        data = prepare_paper_data(pd.read_csv(root/'metrics.csv'),
                                  json.loads((root/'manifest.json').read_text()), allow_pilot, experiment)
    detail = reconstruct_clock_quality(data, pd.read_csv(root/'trace.csv'))
    overall, by_task = clock_summaries(detail), clock_summaries(detail, True)
    output = root/('paper' if data['full'] else 'paper_preview')
    output.mkdir(exist_ok=True)
    detail.to_csv(output/'clock_quality_paired_rows.csv', index=False)
    overall.to_csv(output/'clock_quality_overall.csv', index=False)
    by_task.to_csv(output/'clock_quality_by_task.csv', index=False)
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['budget_overview'])
        fig.subplots_adjust(left=.07, right=.99, bottom=.24, top=.78, wspace=.35)
        definitions = [('available_plans_pct', 'Observed available plans (%)', 100),
                       ('mean_available_value', 'Mean available benefit', 1),
                       ('mean_available_shortfall_pct', 'Mean shortfall to optimum (%)', 100)]
        for ax, (column, label, maximum) in zip(axes, definitions):
            for method in NATIVE:
                rows = overall[overall['Start Method'].eq(method)]
                ax.plot(rows['Elapsed solve wall sec'], rows[column], color=COLORS[method],
                        marker=MARKERS[method], markersize=3, linewidth=1., linestyle=STYLES[method])
            _budget_axis(ax, _budgets(data))
            ax.set_xlabel('Elapsed solve wall time (s)')
            ax.set_ylabel(label)
            ax.set_ylim(-.025*maximum, 1.05*maximum)
            if overall[column].isna().all():
                ax.text(.5,.5,'No certified references',transform=ax.transAxes,
                        ha='center',va='center',fontsize=7)
            _axis(ax)
        fig.legend(handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],linestyle=STYLES[m],
                                   label=LABELS[m]) for m in NATIVE], loc='upper center',
                   ncol=3, fontsize=7, frameon=False)
        _pilot_label(fig,data['full'])
        _save(fig,output,experiment+'14_observed_anytime_quality',png_dpi)
    text = ('Source: one actual '+f'{_benchmark(data):g}-s '+'search per instance and initialization.\n'
            'Only callbacks recorded at or before each elapsed-time point contribute.\n'
            'Validated greedy plans are credited separately at solve time zero.\n'
            'Cold without an observed incumbent is scored separately as idle=0 for available benefit;\n'
            'its observed incumbent remains missing. Missing observed incumbents are not backfilled.\n'
            'Preparation time is excluded. Callbacks document observations, not necessarily the earliest\n'
            'internal discovery time. Certification is counted only after an optimal run finished.\n'
            'Post-hoc gaps use certified references of the same model/objective only.\n'
            'This history is distinct from independent restarts at nominal HiGHS allowances.\n')
    (output/'CLOCK_QUALITY_README.txt').write_text(text)
    if data['full']:
        (output/f'Optional_{experiment}14_in_4_3.tex').write_text(r'''
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{''' + experiment + r'''14_observed_anytime_quality.pdf}
\caption{Experiment ''' + experiment + r''' quality observed within the same ''' + f'{_benchmark(data):g}' + r'''-s
search at each elapsed solve time. Validated starts are available at time zero;
later values use only callbacks recorded by the stated time. The left panel
counts supplied starts or observed solver plans, including zero-benefit plans.
The middle panel averages available value over every instance, scoring idle=0
separately when no plan is observed. The right panel averages paired shortfalls
over certified references only. These are observed histories, distinct from
independent restarts by nominal allowance. Preparation time is excluded.}
\label{fig:exp''' + experiment + r'''-observed-clock}
\end{figure*}
''')
    return {'overall':overall, 'by_task':by_task, 'directory':output}
