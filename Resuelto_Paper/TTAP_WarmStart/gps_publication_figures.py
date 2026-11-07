"""Publication figures from a completed GPS notebook. No solver calls/imports.

Uses METHOD_RESULTS and df_metrics already in the notebook, or exported CSVs.
The style follows the July GPS notebook; data always come from the current run.
"""
from pathlib import Path
from contextlib import contextmanager
import copy
import json
import math
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator, MaxNLocator


DEFAULT_GPS_STYLE = {
    'png_dpi': 600,
    'preview_dpi': 160,
    'formats': ('pdf', 'png'),
    'curve_mode': 'events',  # 'events': original July style; 'step': completion-time jumps.
    'xlim': None,           # Auto: common interval covering all three schedules.
    'benefit_ylim': None,   # Auto: [0, at least 0.70], never truncates the final value.
    'show_grid_allowance': True,
    'show_base_labels': True,
    'task_label_size': 10.5,
    'base_label_size': 8,
    'bar_height': .34,
    'comparison_bar_height': .52,
    'figures': {'gantt': (14., 4.5), 'gantt_comparison': (12.4, 10.2),
                'cumulative': (10., 5.), 'metrics': (12.4, 7.1), 'table': (12.4, 2.8)},
    'colors': {'Greedy Benefit': '#ff7f0e', 'Greedy Density': '#2ca02c', 'HiGHS': '#1f77b4',
               'flight': '#9ecae1', 'service': '#3182bd', 'recovery': '#cccccc',
               'grid_allowance': '#eeeeee'},
    'rc': {'figure.dpi': 160, 'font.family': 'DejaVu Sans', 'font.size': 11,
           'axes.titlesize': 12, 'axes.labelsize': 11, 'xtick.labelsize': 10,
           'ytick.labelsize': 10, 'legend.fontsize': 10, 'axes.linewidth': .8,
           'lines.linewidth': 2., 'lines.markersize': 6., 'lines.antialiased': True,
           'patch.antialiased': True, 'text.antialiased': True,
           'lines.solid_capstyle': 'round', 'lines.dash_capstyle': 'round',
           'lines.solid_joinstyle': 'round', 'lines.dash_joinstyle': 'round',
           'axes.axisbelow': True, 'pdf.fonttype': 42, 'ps.fonttype': 42,
           'svg.fonttype': 'none'},
}


def gps_style(**changes):
    result = copy.deepcopy(DEFAULT_GPS_STYLE)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def _truth(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {'true', '1', '1.0'}:
        return True
    if normalized in {'false', '0', '0.0'}:
        return False
    raise ValueError('Missing or invalid boolean: ' + str(value))


def _helo(value):
    match = re.fullmatch(r'(?:H)?(\d+)(?:\.0)?', str(value).strip(), re.IGNORECASE)
    if not match:
        raise ValueError('Invalid helicopter ID: ' + str(value))
    return int(match.group(1))


@contextmanager
def _paper_context(style):
    rc = {k: v for k, v in style['rc'].items() if k in mpl.rcParams}
    with mpl.rc_context(rc):
        yield


def _axis(ax, axis='both'):
    ax.set_facecolor('white')
    ax.grid(False, which='both', axis='both')
    ax.grid(True, which='major', axis=axis, linestyle='--', linewidth=.65, alpha=.7, color='#bfc3c7')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.8)
        spine.set_edgecolor('black')


def _save(fig, root, name, style):
    root = Path(root)
    formats = tuple(str(x).lower().lstrip('.') for x in style['formats'])
    if not formats or len(set(formats)) != len(formats):
        raise ValueError('Choose at least one format, without duplicates')
    supported = fig.canvas.get_supported_filetypes()
    if any(ext not in supported for ext in formats):
        raise ValueError('Unsupported format; choose from ' + ', '.join(sorted(supported)))
    for key in ['png_dpi', 'preview_dpi']:
        if not math.isfinite(float(style[key])) or float(style[key]) < 72:
            raise ValueError(key + ' must be finite and at least 72')
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    try:
        for extension in formats:
            path = root / (name + '.' + extension)
            # DPI affects raster output or raster elements; all plotted lines/text
            # remain vector objects in the exported PDF and SVG.
            fig.savefig(path, dpi=style['png_dpi'], bbox_inches='tight', facecolor='white')
            result[extension] = path
        preview = root / '_preview'
        preview.mkdir(exist_ok=True)
        result['preview'] = preview / (name + '.png')
        fig.savefig(result['preview'], dpi=style['preview_dpi'], bbox_inches='tight', facecolor='white')
    finally:
        plt.close(fig)
    return result


def _prepare(namespace):
    if 'METHOD_RESULTS' not in namespace:
        raise ValueError('Falta METHOD_RESULTS. Usa el kernel GPS ya ejecutado o load_gps_csv().')
    if 'df_metrics' not in namespace:
        raise ValueError('Falta df_metrics. Ejecuta sólo la celda "Métricas comparativas"; no repitas HiGHS.')
    source = namespace['METHOD_RESULTS']
    if not {'Greedy Benefit', 'Greedy Density'}.issubset(source) or len(source) != 3:
        raise ValueError('Expected both greedy policies and one HiGHS result')
    solver = next(m for m in source if m not in {'Greedy Benefit', 'Greedy Density'})
    methods = ['Greedy Benefit', 'Greedy Density', solver]
    metrics = namespace['df_metrics'].copy(deep=True)
    if metrics['Method'].duplicated().any() or set(metrics['Method']) != set(methods):
        raise ValueError('Metric rows do not match METHOD_RESULTS')
    metrics = metrics.set_index('Method').reindex(methods)
    columns = ['Helo ID', 'Task ID', 'Status', 'Start Time', 'Flight Time',
               'Service Time', 'Completion Time', 'Benefit']
    frames, summaries = {}, {}
    for method in methods:
        original, summary = source[method]
        frame = original.copy(deep=True)
        summary = dict(summary)
        summary['has_incumbent'] = _truth(summary['has_incumbent'])
        summary['optimal'] = _truth(summary['optimal'])
        if (_truth(metrics.loc[method, 'Has Incumbent']) != summary['has_incumbent']
                or _truth(metrics.loc[method, 'Optimal Certified']) != summary['optimal']):
            raise ValueError('Metrics and summary disagree about feasibility or certification')
        if not summary['has_incumbent']:
            if summary['optimal'] or not frame.empty:
                raise ValueError('An absent incumbent has a certificate or a nonempty schedule')
        else:
            if any(c not in frame for c in columns):
                raise ValueError('Missing route columns: ' + ', '.join(c for c in columns if c not in frame))
            for c in ['Start Time', 'Flight Time', 'Service Time', 'Completion Time', 'Benefit']:
                frame[c] = pd.to_numeric(frame[c], errors='raise')
                if not np.isfinite(frame[c]).all():
                    raise ValueError('Non-finite route values in ' + method + ': ' + c)
            frame['Helo ID'] = frame['Helo ID'].map(_helo)
            frame = frame.sort_values(['Helo ID', 'Start Time', 'Completion Time'], kind='stable')
            if (frame[['Start Time', 'Flight Time', 'Service Time', 'Benefit']] < -1e-8).any().any():
                raise ValueError('Negative route times or benefits in ' + method)
            task = frame[frame['Status'].eq('Task Executed')]
            if task['Task ID'].duplicated().any():
                raise ValueError('Duplicate task in ' + method)
            if not np.isclose(task['Benefit'].sum(), summary['benefit'], atol=1e-6, rtol=0):
                raise ValueError('Schedule and solver summary have different benefits')
            if not np.isclose(task['Benefit'].sum(), metrics.loc[method, 'Accumulated Benefit'], atol=1e-6, rtol=0):
                raise ValueError('Schedule and metrics have different benefits')
            if task['Task ID'].nunique() != metrics.loc[method, 'Physical Completed Tasks']:
                raise ValueError('Schedule and metrics have different task counts')
            if not np.isclose(summary['runtime_sec'], metrics.loc[method, 'Runtime sec'], atol=1e-6, rtol=0):
                raise ValueError('Summary and metrics have different runtimes')
            # Keep the exact time-grid remainder visible, rather than extending
            # physical flight/service to fill it or moving a completion event.
            frame['Grid Allowance'] = (frame['Completion Time'] - frame['Start Time']
                                        - frame['Flight Time'] - frame['Service Time'])
            if frame['Grid Allowance'].lt(-1e-7).any():
                raise ValueError('Activity durations exceed their scheduled interval')
            unit = namespace.get('TIME_UNIT')
            if unit is not None and frame['Grid Allowance'].ge(float(unit) + 1e-7).any():
                raise ValueError('Unexpected time-grid remainder')
            for _, group in frame.groupby('Helo ID'):
                starts = group['Start Time'].to_numpy()
                finishes = group['Completion Time'].to_numpy()
                if len(group) > 1 and np.any(starts[1:] < finishes[:-1] - 1e-7):
                    raise ValueError('Overlapping activities in ' + method)
        frames[method], summaries[method] = frame, summary
    if 'helicopters' in namespace:
        helos = sorted(_helo(h) for h in namespace['helicopters'])
    else:
        helos = sorted(set().union(*(set(f['Helo ID']) for f in frames.values() if not f.empty)))
    if not helos:
        raise ValueError('Missing fleet identifiers')
    n_tasks = len(namespace['tasks']) if 'tasks' in namespace else int(namespace['N_TASKS'])
    labels = {m: m for m in methods}
    labels[solver] = 'HiGHS (certified)' if summaries[solver]['optimal'] else 'HiGHS incumbent'
    colors_key = {m: m if m in {'Greedy Benefit', 'Greedy Density'} else 'HiGHS' for m in methods}
    return {'frames': frames, 'summaries': summaries, 'metrics': metrics,
            'methods': methods, 'solver': solver, 'helos': helos, 'n_tasks': n_tasks,
            'labels': labels, 'color_keys': colors_key,
            'scenario': namespace.get('SCENARIO_NAME', 'GPS scenario'),
            'objective_mode': namespace.get('OBJECTIVE_MODE', 'unspecified')}


def _limits(data, style):
    maximum = max((float(f['Completion Time'].max()) for f in data['frames'].values() if not f.empty), default=0.)
    xlim = style['xlim'] or (0., max(10., 5 * math.ceil((maximum + 2) / 5)))
    if xlim[0] > 0 or xlim[1] < maximum - 1e-7 or xlim[1] <= xlim[0]:
        raise ValueError('xlim must cover time zero and every route completion')
    best = max((float(s['benefit']) for s in data['summaries'].values() if s['has_incumbent']), default=0.)
    ylim = style['benefit_ylim'] or (0., max(.70, .05 * math.ceil((best + .04) / .05)))
    if ylim[0] > 0 or ylim[1] < best - 1e-7 or ylim[1] <= ylim[0]:
        raise ValueError('benefit_ylim would truncate an available benefit')
    return tuple(xlim), tuple(ylim)


def _gantt_handles(style):
    values = [('flight', 'Flight'), ('service', 'Task service'), ('recovery', 'Base recovery')]
    handles = [Patch(facecolor=style['colors'][key], edgecolor='none', label=label) for key, label in values]
    if style['show_grid_allowance']:
        handles.append(Patch(facecolor=style['colors']['grid_allowance'], edgecolor='#999999',
                             hatch='///', linewidth=.25, label='Time-grid allowance'))
    return handles


def _draw_gantt(ax, frame, method, data, style, xlim):
    height = style['bar_height']
    for _, row in frame.iterrows():
        h, start = int(row['Helo ID']), float(row['Start Time'])
        flight, service = float(row['Flight Time']), float(row['Service Time'])
        task = row['Status'] == 'Task Executed'
        for left, width, key in [(start, flight, 'flight'), (start + flight, service, 'service' if task else 'recovery')]:
            if width > 0:
                ax.add_patch(Rectangle((left, h - height / 2), width, height,
                    facecolor=style['colors'][key], edgecolor='white', linewidth=.20,
                    antialiased=True, joinstyle='round', zorder=3))
        if row['Grid Allowance'] > 1e-8 and style['show_grid_allowance']:
            ax.add_patch(Rectangle((start + flight + service, h - height / 2), row['Grid Allowance'], height,
                facecolor=style['colors']['grid_allowance'], edgecolor='#999999', linewidth=.2,
                hatch='///', antialiased=True, zorder=2))
        if task and service > 0:
            ax.text(start + flight + service / 2, h, f"T{int(float(row['Task ID']))}",
                ha='center', va='center', fontsize=style['task_label_size'], color='white', zorder=4)
        elif service > 0 and style['show_base_labels']:
            ax.text(start + flight + service / 2, h, 'Base', ha='center', va='center',
                    fontsize=style['base_label_size'], color='#202020', zorder=4)
    if frame.empty:
        message = 'No solver incumbent' if not data['summaries'][method]['has_incumbent'] else 'Feasible plan without activities'
        ax.text(.5, .5, message, transform=ax.transAxes, ha='center', va='center')
    ax.set_yticks(data['helos'], ['H' + str(h) for h in data['helos']])
    ax.set(xlim=xlim, ylim=(min(data['helos']) - .8, max(data['helos']) + .8), ylabel='Helicopter')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_title(data['labels'][method], loc='left', pad=10)
    _axis(ax, 'x')


def _events(frame, end):
    tasks = frame[frame['Status'].eq('Task Executed')]
    points = tasks.groupby('Completion Time')['Benefit'].sum().sort_index()
    x = np.r_[0., points.index.to_numpy(dtype=float)]
    y = np.r_[0., points.cumsum().to_numpy(dtype=float)]
    return x, y, np.r_[x, end], np.r_[y, float(points.sum())]


def _draw_curve(ax, frame, method, data, style, xlim):
    if not data['summaries'][method]['has_incumbent']:
        return None
    x, y, x_end, y_end = _events(frame, xlim[1])
    key, index = data['color_keys'][method], data['methods'].index(method)
    color = style['colors'][key]
    label = data['labels'][method] + f" ({data['summaries'][method]['benefit']:.4f})"
    marker = ['o', 's', 'o'][index]
    # The dashed Density line and smaller open squares keep identical greedy
    # histories visible without artificially shifting their values or times.
    options = {'color': color, 'linewidth': 2., 'linestyle': '--' if index == 1 else '-', 'label': label}
    if style['curve_mode'] == 'step':
        line, = ax.step(x_end, y_end, where='post', **options)
    elif style['curve_mode'] == 'events':
        line, = ax.plot(x_end, y_end, **options)
    else:
        raise ValueError("curve_mode must be 'events' or 'step'")
    ax.plot(x[1:], y[1:], linestyle='none', color=color, marker=marker,
            markersize=4.3 if index == 1 else 6., markerfacecolor='white',
            markeredgewidth=1.25, label='_nolegend_', zorder=4)
    return line


def _curve_axis(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim, xlabel='Mission time (min)', ylabel='Cumulative normalized benefit')
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(.05))
    _axis(ax)


def _metric_figures(data, root, style):
    table = data['metrics']
    labels = ['Greedy\nBenefit', 'Greedy\nDensity', 'HiGHS']
    colors = [style['colors'][data['color_keys'][m]] for m in data['methods']]
    definitions = [('Accumulated Benefit', 'Normalized benefit', '{:.4f}'),
                   ('Physical Completed Tasks', 'Completed tasks', '{:.0f}'),
                   ('Operationally Relevant Tasks', r'Tasks with $S_i \geq 0.5$', '{:.0f}'),
                   ('Average Response Time', 'Mean task completion time (min)', '{:.1f}'),
                   ('Total Flight Time', 'Total flight time (min)', '{:.1f}'),
                   ('Runtime sec', 'Recorded runtime (s; log scale)', '{:.3g}')]
    fig, axes = plt.subplots(2, 3, figsize=style['figures']['metrics'])
    fig.subplots_adjust(left=.075, right=.98, bottom=.12, top=.94, hspace=.65, wspace=.32)
    for index, (ax, (column, label, fmt)) in enumerate(zip(axes.flat, definitions)):
        values = pd.to_numeric(table[column], errors='raise').to_numpy(dtype=float)
        valid = np.isfinite(values)
        if column == 'Runtime sec' and np.any(values[valid] <= 0):
            raise ValueError('Positive recorded runtimes are required for the log-scale panel')
        x = np.arange(len(labels))
        bars = ax.bar(x[valid], values[valid], color=np.array(colors)[valid], width=.58,
                       edgecolor='white', linewidth=.6, zorder=3)
        for j in x[~valid]:
            ax.text(j, .06, 'n/a', transform=ax.get_xaxis_transform(), ha='center')
        for bar, value in zip(bars, values[valid]):
            text = (f'{int(value)}/{data["n_tasks"]}' if column == 'Physical Completed Tasks' else fmt.format(value))
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 5),
                        textcoords='offset points', ha='center', va='bottom', fontsize=10)
        ax.set_xticks(x, labels)
        ax.set_xlim(-.65, len(labels) - .35)
        ax.set_title(f'({chr(97+index)}) {label}', fontsize=11, pad=13)
        if column == 'Runtime sec':
            ax.set_yscale('log')
            if valid.any():
                ax.set_ylim(float(values[valid].min()) * .4, float(values[valid].max()) * 3.)
        else:
            maximum = (data['n_tasks'] if 'Tasks' in column else max(.1, values[valid].max()) if valid.any() else 1.)
            ax.set_ylim(0, maximum * 1.22)
            if 'Tasks' in column:
                ax.set_ylim(0, data['n_tasks'] * 1.08)
                ax.yaxis.set_major_locator(MultipleLocator(5 if data['n_tasks'] >= 20 else 2))
        _axis(ax, 'y')
    metrics_path = _save(fig, root, 'GPS_metrics_final', style)
    # A publication table is often easier to use than six bars in a short paper.
    shortfall = pd.to_numeric(table['Optimality Gap vs Exact'], errors='raise') * 100
    rows = []
    for method in data['methods']:
        r = table.loc[method]
        present = data['summaries'][method]['has_incumbent']
        rows.append([data['labels'][method], f"{r['Accumulated Benefit']:.4f}" if present else '--',
            f"{int(r['Physical Completed Tasks'])}/{data['n_tasks']}" if present else '--',
            f"{int(r['Operationally Relevant Tasks'])}" if present else '--',
            f"{r['Runtime sec']:.4f}", f'{shortfall.loc[method]:.2f}' if pd.notna(shortfall.loc[method]) else '--'])
    headers = ['Method', 'Normalized\nbenefit', 'Completed\ntasks', 'Tasks with\n$S_i \\geq 0.5$',
               'Runtime\n(s)', 'Shortfall to\noptimum (%)']
    fig, ax = plt.subplots(figsize=style['figures']['table'])
    ax.axis('off')
    artist = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                      colWidths=[.25, .15, .15, .15, .15, .15], bbox=[0, .22, 1, .68])
    artist.auto_set_font_size(False)
    artist.set_fontsize(10)
    for (row, col), cell in artist.get_celld().items():
        cell.set_edgecolor('#c9cdd2')
        cell.set_linewidth(.55)
        if row == 0:
            cell.set_facecolor('#edf1f5')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('white' if row % 2 else '#f8f9fa')
            if col == 0:
                cell.set_text_props(color=colors[row - 1])
    ax.text(0, .08, 'Fixed GPS instance; no repeated-instance SD. Shortfalls require a certified reference.',
            transform=ax.transAxes, fontsize=9, ha='left')
    fig.tight_layout()
    table_path = _save(fig, root, 'GPS_metrics_table', style)
    return metrics_path, table_path


def export_gps_figures(namespace, style=None, output_directory=None, show=True):
    """Call from an executed notebook: export_gps_figures(globals(), style=GPS_STYLE)."""
    style = gps_style() if style is None else copy.deepcopy(style)
    data = _prepare(namespace)
    root = (Path(output_directory) if output_directory is not None
            else Path(namespace.get('FIGURES_DIR', 'figures_ttap_GPS')) / 'paper_restyled')
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    xlim, ylim = _limits(data, style)
    figures = {}
    with _paper_context(style):
        for index, method in enumerate(data['methods']):
            suffix = ['greedy_benefit', 'greedy_density', 'highs'][index]
            fig, ax = plt.subplots(figsize=style['figures']['gantt'])
            fig.subplots_adjust(left=.06, right=.99, bottom=.15, top=.80)
            _draw_gantt(ax, data['frames'][method], method, data, style, xlim)
            ax.set_xlabel('Mission time (min)')
            fig.legend(handles=_gantt_handles(style), loc='upper center', bbox_to_anchor=(.5, .99),
                       ncol=4 if style['show_grid_allowance'] else 3, frameon=False)
            figures['gantt_' + suffix] = _save(fig, root, 'GPS_gantt_' + suffix, style)
        fig, axes = plt.subplots(3, 1, figsize=style['figures']['gantt_comparison'], sharex=True, sharey=True)
        fig.subplots_adjust(left=.06, right=.99, bottom=.07, top=.91, hspace=.48)
        comparison_style = copy.deepcopy(style)
        comparison_style['bar_height'] = style['comparison_bar_height']
        for ax, method in zip(axes, data['methods']):
            _draw_gantt(ax, data['frames'][method], method, data, comparison_style, xlim)
        axes[-1].set_xlabel('Mission time (min)')
        fig.legend(handles=_gantt_handles(style), loc='upper center', bbox_to_anchor=(.5, .99),
                   ncol=4 if style['show_grid_allowance'] else 3, frameon=False)
        figures['gantt_comparison'] = _save(fig, root, 'GPS_gantt_comparison_three_methods', style)
        for index, method in enumerate(data['methods']):
            suffix = ['greedy_benefit', 'greedy_density', 'highs'][index]
            fig, ax = plt.subplots(figsize=style['figures']['cumulative'])
            _draw_curve(ax, data['frames'][method], method, data, style, xlim)
            _curve_axis(ax, xlim, ylim)
            ax.set_title(data['labels'][method], pad=10)
            if not data['summaries'][method]['has_incumbent']:
                ax.text(.5, .5, 'No solver incumbent', transform=ax.transAxes, ha='center')
            fig.tight_layout()
            figures['cumulative_' + suffix] = _save(fig, root, 'GPS_cumulative_' + suffix, style)
        fig, ax = plt.subplots(figsize=style['figures']['cumulative'])
        for method in data['methods']:
            _draw_curve(ax, data['frames'][method], method, data, style, xlim)
        _curve_axis(ax, xlim, ylim)
        ax.legend(loc='upper left', frameon=True, framealpha=.95)
        gx, gy, _, _ = _events(data['frames']['Greedy Benefit'], xlim[1])
        dx, dy, _, _ = _events(data['frames']['Greedy Density'], xlim[1])
        coincident = np.array_equal(gx, dx) and np.allclose(gy, dy, atol=1e-10, rtol=0)
        if coincident:
            ax.text(.98, .045, 'Greedy Benefit and Density coincide in cumulative benefit',
                    transform=ax.transAxes, ha='right', fontsize=9,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
        fig.tight_layout()
        figures['cumulative_comparison'] = _save(fig, root, 'GPS_cumulative_comparison_three_methods', style)
        figures['metrics'], figures['metrics_table'] = _metric_figures(data, root, style)
    # Export copies and plotted event coordinates; do not rewrite original CSVs.
    data['metrics'].reset_index().to_csv(root / 'GPS_metrics_source_copy.csv', index=False)
    events, timing = [], []
    for method, frame in data['frames'].items():
        if not frame.empty:
            selected = frame[frame['Status'].eq('Task Executed')]
            p = selected.groupby('Completion Time')['Benefit'].sum().sort_index()
            events.extend({'Method': method, 'Completion Time': float(t), 'Cumulative Benefit': float(y)}
                          for t, y in zip(p.index, p.cumsum()))
            timing.append(frame.assign(Method=method))
    pd.DataFrame(events, columns=['Method', 'Completion Time', 'Cumulative Benefit']).to_csv(root / 'GPS_cumulative_events.csv', index=False)
    if timing:
        pd.concat(timing, ignore_index=True).to_csv(root / 'GPS_drawn_schedule_intervals.csv', index=False)
    _captions(root, style)
    manifest = {'scenario': data['scenario'], 'objective_mode': data['objective_mode'],
        'methods': data['methods'], 'helos': data['helos'], 'n_tasks': data['n_tasks'],
        'benefits': {m: float(s['benefit']) if s['has_incumbent'] else None for m, s in data['summaries'].items()},
        'optimal_certified': {m: s['optimal'] for m, s in data['summaries'].items()},
        'coincident_greedy_cumulative': bool(coincident), 'style': style,
        'scope': 'Presentation only; existing starts, completions, benefits and metrics are unchanged.',
        'grid_allowance': 'Completion Time minus Start Time, Flight Time and Service Time.',
        'events': 'Benefit credited at recorded task completions; connecting lines are visual guides.'}
    (root / 'GPS_graphics_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if show:
        from IPython.display import display, Image
        print('Figuras y métricas guardadas en:', root)
        for key in ['gantt_comparison', 'cumulative_comparison', 'metrics', 'metrics_table']:
            display(Image(filename=str(figures[key]['preview'])))
    return {'directory': root, 'figures': figures, 'metrics': data['metrics'].copy(), 'manifest': manifest}


def _captions(root, style):
    if 'pdf' not in style['formats']:
        return
    line_note = ('Points show cumulative values at task completions; connecting lines are visual guides.'
                 if style['curve_mode'] == 'events' else 'Benefit is credited at recorded task completion times.')
    grid_note = ('Hatched segments show the time-grid allowance after physical flight and service.'
                 if style['show_grid_allowance'] else 'Unfilled gaps can contain time-grid allowances.')
    text = r'''% Needs graphicx. Select the figures appropriate for the paper.
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{GPS_gantt_comparison_three_methods.pdf}
\caption{Schedules for the fixed GPS scenario under Greedy Benefit, Greedy Density
and HiGHS. All panels use the same mission-time interval and fleet identifiers.
Recorded activity starts, physical flight and service durations, and completion
times are preserved. ''' + grid_note + r'''}
\label{fig:gps-gantt-three-methods}
\end{figure*}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\columnwidth]{GPS_cumulative_comparison_three_methods.pdf}
\caption{Cumulative normalized benefit in the fixed GPS scenario. ''' + line_note + r'''
The two greedy histories can coincide even when their flight routes differ.
This is one fixed instance; no repeated-instance uncertainty bands are shown.}
\label{fig:gps-cumulative-three-methods}
\end{figure}

% Optional descriptive metrics; they do not share a common objective scale.
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{GPS_metrics_final.pdf}
\caption{Descriptive outcomes of the fixed GPS scenario. Task counts, completion
times, flight totals and recorded runtimes summarize the existing schedules.
HiGHS maximizes temporal benefit; the remaining metrics are not independent
optimization objectives. Runtime uses a logarithmic scale. No repeated-instance
error bars are inferred from this single scenario.}
\label{fig:gps-metrics}
\end{figure*}
'''
    (root / 'GPS_figures_overleaf.tex').write_text(text, encoding='utf-8')


def load_gps_csv(directory):
    """Optional recovery if the original kernel is closed; reads existing exports."""
    root = Path(directory)
    routes = pd.read_csv(root / 'talcahuano_routes.csv')
    comparison = pd.read_csv(root / 'talcahuano_comparison.csv')
    metrics = pd.read_csv(root / 'talcahuano_metrics.csv')
    balance = pd.read_csv(root / 'talcahuano_helo_balance.csv')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    n_match = re.search(r'(\d+)tasks', manifest.get('scenario', ''))
    if not n_match:
        raise ValueError('Cannot recover task count from the saved scenario name')
    results = {}
    for _, row in comparison.iterrows():
        method = row['Method']
        summary = {'has_incumbent': _truth(row['Has Incumbent']), 'optimal': _truth(row['Optimal Certified']),
            'benefit': row['Accumulated Benefit'], 'runtime_sec': row['Runtime sec']}
        results[method] = (routes[routes['Method'].eq(method)].copy(), summary)
    return {'METHOD_RESULTS': results, 'df_metrics': metrics, 'FIGURES_DIR': root,
        'SCENARIO_NAME': manifest['scenario'], 'OBJECTIVE_MODE': manifest['objective_mode'],
        'TIME_UNIT': manifest['time_unit_minutes'], 'N_TASKS': int(n_match.group(1)),
        'helicopters': {h: {} for h in balance['Helo ID'].drop_duplicates()}}
