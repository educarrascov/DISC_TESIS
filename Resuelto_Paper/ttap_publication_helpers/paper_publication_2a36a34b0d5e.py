"""Replot completed TTAP experiments. No solver, generator, or frozen runtime import.

Three samples are deliberately distinct:
* main quality: a matched certified cohort at the longest configured limit;
* configured-limit results: all instances, independent solver restarts;
* observed history: all instances, callbacks from one long search per start.
"""
from pathlib import Path
from contextlib import contextmanager
from urllib.request import Request, urlopen
import hashlib
import json
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

DATA_REVISION = '2cd8035c8417b3c4f5107c57ee174beef046ced3'
REPOSITORY = 'educarrascov/DISC_TESIS'
RUNS = {
    'A': {'directory': 'A_paper_full_997dc6f9b016', 'files': {
        'manifest.json': '19a2535d7e41ac45bf6c3606ff47c2ebea4a97a9',
        'metrics.csv': '97fcea06a3296b398b69912eb62316ea5a9ac134',
        'trace.csv': '00af63c26d3fcd28a3dbd3676322c58cffc53f73'}},
    'B': {'directory': 'B_paper_full_a1dfd983cde0', 'files': {
        'manifest.json': 'ac56092b2d7518f50c2d8643c1faffc553c9e469',
        'metrics.csv': 'ae017e243cb5f9742a02c464d509c992a224e221',
        'trace.csv': '992ca1737205895523a64b79c6e168b14686d444'}}}
METHODS = ('Cold', 'GBStart', 'GDStart')
HEURISTICS = ('GreedyBenefit', 'GreedyDensity')
KEYS = ['Scenario', 'Tasks', 'Seed']
LABELS = {'Cold': 'HiGHS cold', 'GBStart': 'HiGHS + Benefit start',
          'GDStart': 'HiGHS + Density start', 'GreedyBenefit': 'Greedy Benefit',
          'GreedyDensity': 'Greedy Density', 'Certified': 'HiGHS certified optimum'}
DEFAULT_STYLE = {
    'png_dpi': 600, 'preview_dpi': 160,
    'colors': {'Cold': '#1f77b4', 'GBStart': '#9467bd', 'GDStart': '#17becf',
               'GreedyBenefit': '#ff7f0e', 'GreedyDensity': '#2ca02c',
               'Certified': '#1f77b4', 'A': '#0072B2', 'B': '#D55E00'},
    'markers': {'Cold': 'o', 'GBStart': 's', 'GDStart': '^',
                'GreedyBenefit': 'o', 'GreedyDensity': 'o', 'Certified': 'o'},
    'linestyles': {'Cold': '-', 'GBStart': '--', 'GDStart': '-.',
                   'GreedyBenefit': '-', 'GreedyDensity': '-', 'Certified': '-'},
    'rc': {'figure.dpi': 160, 'font.family': 'DejaVu Sans', 'font.size': 10,
           'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
           'ytick.labelsize': 9, 'legend.fontsize': 8.5, 'lines.linewidth': 1.8,
           'lines.markersize': 6, 'axes.linewidth': .8, 'pdf.fonttype': 42,
           'ps.fonttype': 42, 'svg.fonttype': 'none', 'lines.antialiased': True,
           'patch.antialiased': True, 'text.antialiased': True,
           'lines.solid_capstyle': 'round', 'lines.dash_capstyle': 'round',
           'lines.solid_joinstyle': 'round', 'lines.dash_joinstyle': 'round',
           'axes.axisbelow': True},
    'sizes': {'quality': (9., 5.), 'runtime': (8.4, 5.), 'effort': (12.4, 4.9),
              'by_task': (14., 4.2), 'overview': (12.4, 4.4),
              'combined': (12.4, 4.7), 'dashboard': (12.4, 7.6)},
    'quality_ylim': None,
    'available_ylim': (-.015, 1.015),
}


def style_config(**changes):
    """Return an independent, editable style dictionary."""
    value = json.loads(json.dumps(DEFAULT_STYLE))
    for key, setting in changes.items():
        if isinstance(setting, dict) and isinstance(value.get(key), dict):
            value[key].update(setting)
        else:
            value[key] = setting
    return value


@contextmanager
def paper_style(style):
    rc = {key: value for key, value in style['rc'].items() if key in mpl.rcParams}
    with plt.style.context('seaborn-v0_8-whitegrid'), mpl.rc_context(rc):
        yield


def _boolean(values, name):
    if pd.api.types.is_bool_dtype(values) and values.notna().all():
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    converted = normalized.map({'true': True, 'false': False, '1': True, '0': False,
                                '1.0': True, '0.0': False})
    if converted.isna().any():
        raise ValueError('Missing or invalid boolean in ' + name)
    return converted.astype(bool)


def _blob_sha(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def download_published(experiment, cache_directory):
    """Download six small source files in total; verify fixed Git blob identities.

    Local experiment folders need no download: pass them directly to load_run.
    This cache is separate from every running kernel's verified runtime.
    """
    experiment = experiment.upper()
    info = RUNS[experiment]
    root = Path(cache_directory).expanduser().resolve() / info['directory']
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in info['files'].items():
        target = root / name
        if target.is_file() and _blob_sha(target.read_bytes()) == expected:
            continue
        url = (f'https://raw.githubusercontent.com/{REPOSITORY}/{DATA_REVISION}/'
               f'Resuelto_Paper/ttap_runs_paper_2000dpi/{info["directory"]}/{name}')
        request = Request(url, headers={'User-Agent': 'TTAP-publication-figures/1.0'})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if _blob_sha(content) != expected:
            raise RuntimeError('Published data did not match the fixed revision: ' + name)
        partial = target.with_suffix(target.suffix + '.download')
        partial.write_bytes(content)
        os.replace(partial, target)
    return root


def load_run(directory, experiment=None, require_full=True):
    """Validate the paired design, objective, certification, and supplied starts."""
    root = Path(directory).expanduser().resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    exp = str(config['experiment']).upper()
    if experiment is not None and exp != experiment.upper():
        raise ValueError('Selected folder belongs to Experiment ' + exp)
    tasks, seeds = sorted(config['tasks']), sorted(config['seeds'])
    budgets = sorted(float(x) for x in config['budgets'])
    if require_full and (tasks != [20, 30, 40, 50, 60] or seeds != list(range(1, 31))):
        raise ValueError('The full paper requires all five sizes and 30 seeds')
    if budgets != ([1., 5., 15., 60., 600.] + ([1800.] if exp == 'B' else [])):
        raise ValueError('Unexpected budget grid; review the protocol before pooling')
    raw = pd.read_csv(root / 'metrics.csv')
    warm = raw[raw['Study Stage'].eq('Warm')].copy()
    heur = raw[raw['Study Stage'].eq('Heuristic')].copy()
    expected = {(t, s, b, m) for t in tasks for s in seeds for b in budgets for m in METHODS}
    actual = set(warm[['Tasks', 'Seed', 'Time Limit sec', 'Start Method']].itertuples(index=False, name=None))
    if actual != expected or len(warm) != len(expected):
        raise ValueError('Incomplete or duplicated solver grid; do not pool partial runs')
    expected_h = {(t, s, m) for t in tasks for s in seeds for m in ('GBStart', 'GDStart')}
    actual_h = set(heur[['Tasks', 'Seed', 'Start Method']].itertuples(index=False, name=None))
    if actual_h != expected_h or len(heur) != len(expected_h):
        raise ValueError('Missing or duplicate matched greedy baselines')
    if len(raw) != len(warm) + len(heur):
        raise ValueError('Unrecognized study stages; review before pooling')
    if raw['Objective SHA256'].isna().any() or raw['Objective SHA256'].nunique() != 1:
        raise ValueError('Mixed objective fingerprints')
    if not raw['Objective Mode'].eq(config['objective']['mode']).all():
        raise ValueError('Objective mode disagrees with the run manifest')
    for name in ['Has Incumbent', 'Solver Has Incumbent', 'Optimal Certified', 'Warm Start Submitted']:
        warm[name] = _boolean(warm[name], name)
    for name in ['Model SHA256', 'Scenario SHA256', 'Variables', 'Constraints']:
        groups = warm.groupby(KEYS)[name]
        if warm[name].isna().any() or groups.nunique().ne(1).any():
            raise ValueError('Changed model between paired treatments: ' + name)
    for name in ['HiGHS Version', 'Threads', 'Random Seed', 'MIP Relative Tolerance']:
        if warm[name].isna().any() or warm[name].nunique() != 1:
            raise ValueError('Mixed solver settings: ' + name)
    for frame in [warm, heur]:
        frame['Value'] = pd.to_numeric(frame['Normalized Mission Value'], errors='raise')
        if ((frame['Value'] < -1e-8) | (frame['Value'] > 1 + 1e-8)).any():
            raise ValueError('Normalized benefit outside [0, 1]')
    if warm.loc[warm['Has Incumbent'], 'Value'].isna().any() or heur['Value'].isna().any():
        raise ValueError('Available plan without an objective value')
    warm['Available'] = warm['Value'].fillna(0.)
    warm['Solver value'] = pd.to_numeric(warm['Solver Incumbent Value'], errors='raise').where(warm['Solver Has Incumbent'])
    if warm.loc[warm['Solver Has Incumbent'], 'Solver value'].isna().any():
        raise ValueError('Solver incumbent without an objective value')
    if (warm['Optimal Certified'] & ~warm['Solver Has Incumbent']).any():
        raise ValueError('Optimal status without a solver incumbent')
    cert = warm[warm['Optimal Certified']]
    refs = cert.groupby(KEYS, as_index=False)['Value'].max().rename(columns={'Value': 'Optimum'})
    spread = cert.groupby(KEYS)['Value'].agg(lambda x: x.max() - x.min())
    if spread.gt(2e-6).any():
        raise ValueError('Certified objectives disagree beyond the reporting tolerance')
    warm = warm.merge(refs, on=KEYS, how='left', validate='many_to_one')
    heur = heur.merge(refs, on=KEYS, how='left', validate='many_to_one')
    if (warm['Available'] > warm['Optimum'] + 2e-6).any() or (heur['Value'] > heur['Optimum'] + 2e-6).any():
        raise ValueError('Feasible value exceeds a certified reference')
    warm['Available gap pct'] = (100 * (warm['Optimum'] - warm['Available']) / warm['Optimum']).clip(lower=0)
    warm['Solver gap pct'] = (100 * (warm['Optimum'] - warm['Solver value']) / warm['Optimum']).clip(lower=0)
    baseline = heur.pivot(index=KEYS, columns='Start Method', values='Value')
    for method in ('GBStart', 'GDStart'):
        group = warm[warm['Start Method'].eq(method)].merge(
            baseline[[method]].reset_index(), on=KEYS, validate='many_to_one')
        if not group['Warm Start Submitted'].all():
            raise ValueError('A warm run did not submit its initial solution')
        if not np.allclose(group[method], group['Warm Start Value'], atol=1e-8, rtol=0):
            raise ValueError('Supplied start differs from its matched greedy')
        if (group['Available'] < group[method] - 1e-8).any():
            raise ValueError('A validated initial plan was lost')
    warm['Overrun sec'] = warm['Solver Time sec'] - warm['Time Limit sec']
    trace = pd.read_csv(root / 'trace.csv')
    if not trace['Objective SHA256'].eq(raw['Objective SHA256'].iloc[0]).all():
        raise ValueError('Callback data use another objective')
    return {'directory': root, 'experiment': exp, 'manifest': manifest, 'raw': raw,
            'warm': warm, 'heuristics': heur, 'references': refs, 'trace': trace,
            'tasks': tasks, 'seeds': seeds, 'budgets': budgets, 'benchmark': max(budgets),
            'n': len(tasks) * len(seeds)}


def quality_data(study, cohort='Cold'):
    """Use one identical certified cohort for HiGHS and both heuristics.

    Cold = original paper's certified cold runs; common = all three certified;
    reference = any certified run of the same model. No uncertified value is
    presented as an optimum under any of these choices.
    """
    bench = study['warm'][study['warm']['Time Limit sec'].eq(study['benchmark'])]
    if cohort == 'Cold':
        selected = bench[bench['Start Method'].eq('Cold') & bench['Optimal Certified']][KEYS]
    elif cohort == 'common':
        flags = bench.groupby(KEYS)['Optimal Certified'].all().reset_index()
        selected = flags[flags['Optimal Certified']][KEYS]
    elif cohort == 'reference':
        selected = study['references'][KEYS]
    else:
        raise ValueError("quality_cohort must be 'Cold', 'common', or 'reference'")
    certified = bench[bench['Optimal Certified']].sort_values(['Scenario', 'Start Method'])
    if cohort == 'Cold':
        certified = certified[certified['Start Method'].eq('Cold')]
    certified = certified.drop_duplicates(KEYS).merge(selected, on=KEYS, validate='one_to_one')
    certified['Plot method'] = 'Certified'
    selected_heur = study['heuristics'].merge(selected, on=KEYS, validate='many_to_one')
    selected_heur['Plot method'] = selected_heur['Start Method'].map(
        {'GBStart': 'GreedyBenefit', 'GDStart': 'GreedyDensity'})
    combined = pd.concat([certified, selected_heur], ignore_index=True)
    stats = combined.groupby(['Tasks', 'Plot method'], as_index=False).agg(
        n=('Scenario', 'size'), mean=('Value', 'mean'), sd=('Value', 'std'),
        tasks_mean=('Physical Completed Tasks', 'mean'), tasks_sd=('Physical Completed Tasks', 'std'))
    coverage = pd.DataFrame({'Tasks': study['tasks']})
    coverage['Generated'] = len(study['seeds'])
    coverage = coverage.merge(selected.groupby('Tasks').size().rename('Certified cohort'),
                              left_on='Tasks', right_index=True, how='left')
    coverage['Certified cohort'] = coverage['Certified cohort'].fillna(0).astype(int)
    optimum = stats[stats['Plot method'].eq('Certified')].set_index('Tasks')['mean']
    stats['Mean-ratio shortfall pct'] = (100 * (1 - stats['mean'] / stats['Tasks'].map(optimum))).clip(lower=0)
    return stats, coverage, combined


def budget_data(study, by_task=False):
    keys = (['Tasks'] if by_task else []) + ['Start Method', 'Time Limit sec']
    result = study['warm'].groupby(keys, as_index=False).agg(
        n=('Scenario', 'size'), certified=('Optimal Certified', 'sum'),
        solver_incumbents=('Solver Has Incumbent', 'sum'),
        mean_incumbent=('Solver value', 'mean'), median_incumbent=('Solver value', 'median'),
        mean_available=('Available', 'mean'), known_references=('Optimum', 'count'),
        gap_n=('Solver gap pct', 'count'), mean_gap_pct=('Solver gap pct', 'mean'),
        median_gap_pct=('Solver gap pct', 'median'), mean_available_gap_pct=('Available gap pct', 'mean'),
        mean_solve_seconds=('Solver Time sec', 'mean'), max_solve_seconds=('Solver Time sec', 'max'))
    result['feasible_uncertified'] = result['solver_incumbents'] - result['certified']
    result['no_incumbent'] = result['n'] - result['solver_incumbents']
    for col in ['certified', 'solver_incumbents']:
        result[col + '_pct'] = 100 * result[col] / result['n']
    if not (result['certified'] + result['feasible_uncertified'] + result['no_incumbent']).eq(result['n']).all():
        raise AssertionError('Status categories do not partition the cohort')
    return result


def clock_data(study, points=None):
    """Reconstruct observations, including starts at t=0; never backfill a final value."""
    points = study['budgets'] if points is None else sorted(set(float(p) for p in points))
    if not points or min(points) < 0 or max(points) > study['benchmark']:
        raise ValueError('Clock points must lie within the source run allowance')
    bench = study['warm'][study['warm']['Time Limit sec'].eq(study['benchmark'])]
    trace = study['trace']
    trace = trace[trace['Study Stage'].eq('Warm') & trace['Budget sec'].eq(study['benchmark'])]
    groups = {key: g.sort_values('Elapsed solve wall sec', kind='stable')
              for key, g in trace.groupby(['Scenario', 'Start Method'])}
    expected = set(bench[['Scenario', 'Start Method']].itertuples(index=False, name=None))
    if set(groups) != expected:
        raise ValueError('Missing or extra long-search histories')
    rows = []
    for _, run in bench.iterrows():
        method = run['Start Method']
        events = groups[(run['Scenario'], method)]
        time = pd.to_numeric(events['Elapsed solve wall sec'], errors='raise').to_numpy()
        values = pd.to_numeric(events['Solver Incumbent Value'], errors='raise').to_numpy()
        expected_initial = 'Idle fallback available' if method == 'Cold' else 'Validated start available'
        if (events['Event'].eq(expected_initial).sum() != 1 or time[0] != 0
                or events.iloc[0]['Event'] != expected_initial or events['Event'].eq('Solve finished').sum() != 1):
            raise ValueError('Incomplete callback history: ' + run['Scenario'] + '/' + method)
        finish = events[events['Event'].eq('Solve finished')].iloc[0]
        if (not np.isfinite(time).all() or (time < 0).any()
                or not np.isclose(finish['Elapsed solve wall sec'], run['Solver Time sec'], atol=1e-6, rtol=0)
                or not np.isclose(finish['Available Value'], run['Available'], atol=1e-6, rtol=0)
                or not events['Tasks'].eq(run['Tasks']).all() or not events['Seed'].eq(run['Seed']).all()):
            raise ValueError('Callback history does not match its long run')
        initial = 0. if method == 'Cold' else float(run['Warm Start Value'])
        for point in points:
            seen = values[(time <= point) & np.isfinite(values)]
            observed = seen.max() if len(seen) else np.nan
            available = max(initial, observed if np.isfinite(observed) else 0.)
            ref = run['Optimum']
            rows.append({**{key: run[key] for key in KEYS}, 'Start Method': method,
                'Elapsed solve wall sec': point, 'Available': available,
                'Solver value': observed, 'Observed solver plan': bool(len(seen)),
                'Available plan': method != 'Cold' or bool(len(seen)), 'Optimum': ref,
                'Available gap pct': 100 * max(0., ref - available) / ref if pd.notna(ref) else np.nan,
                'Certified by time': bool(run['Optimal Certified'] and run['Solver Time sec'] <= point)})
    return pd.DataFrame(rows)


def clock_summary(detail):
    result = detail.groupby(['Start Method', 'Elapsed solve wall sec'], as_index=False).agg(
        n=('Scenario', 'size'), mean_available=('Available', 'mean'),
        mean_observed_incumbent=('Solver value', 'mean'),
        median_available=('Available', 'median'), solver_incumbents=('Observed solver plan', 'sum'),
        available_plans=('Available plan', 'sum'), known_references=('Optimum', 'count'),
        mean_available_gap_pct=('Available gap pct', 'mean'), certified=('Certified by time', 'sum'))
    result['available_plans_pct'] = 100 * result['available_plans'] / result['n']
    result['solver_incumbents_pct'] = 100 * result['solver_incumbents'] / result['n']
    result['certified_pct'] = 100 * result['certified'] / result['n']
    return result


def attribution_data(study):
    cold = study['warm'][study['warm']['Start Method'].eq('Cold')][
        KEYS + ['Time Limit sec', 'Available']].rename(columns={'Available': 'Cold available'})
    detail = study['warm'][study['warm']['Start Method'].isin(('GBStart', 'GDStart'))].merge(
        cold, on=KEYS + ['Time Limit sec'], validate='many_to_one')
    detail['Greedy gain'] = detail['Available'] - detail['Warm Start Value']
    detail['Injection gain'] = detail['Available'] - np.maximum(detail['Cold available'], detail['Warm Start Value'])
    detail['Injection win'] = detail['Injection gain'] > 1e-6
    detail['Injection loss'] = detail['Injection gain'] < -1e-6
    result = detail.groupby(['Start Method', 'Time Limit sec'], as_index=False).agg(
        n=('Scenario', 'size'), mean_greedy_gain=('Greedy gain', 'mean'),
        mean_injection_gain=('Injection gain', 'mean'), median_injection_gain=('Injection gain', 'median'),
        wins=('Injection win', 'sum'), losses=('Injection loss', 'sum'))
    result['ties'] = result['n'] - result['wins'] - result['losses']
    return result, detail


def _axis(ax, grid='both'):
    ax.grid(True, axis=grid, linestyle='--', linewidth=.6, alpha=.7, color='#C7C7C7')
    ax.grid(False, which='minor')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(.8)


def _budget_axis(ax, points):
    ax.set_xscale('log')
    ax.set_xticks(points, [f'{x:g}' for x in points], rotation=35)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(points) * .75, max(points) * 1.3)
    _axis(ax, 'y')


def _line(ax, x, y, method, style, label=None, **kwargs):
    return ax.plot(x, y, color=style['colors'][method], marker=style['markers'][method],
                   linestyle=style['linestyles'][method], label=label or LABELS[method], **kwargs)


def _legend(fig, style, methods=METHODS, top=.98):
    handles = [Line2D([], [], color=style['colors'][m], marker=style['markers'][m],
                      linestyle=style['linestyles'][m], label=LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, top),
               ncol=len(methods), frameon=False)


def _save(fig, folder, name, style):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dpi = float(style['png_dpi'])
    if not math.isfinite(dpi) or dpi < 72:
        raise ValueError('png_dpi must be finite and at least 72')
    paths = {}
    try:
        for suffix in ['pdf', 'svg']:
            path = folder / (name + '.' + suffix)
            fig.savefig(path, bbox_inches='tight', facecolor='white')
            paths[suffix] = path
        fig.savefig(folder / (name + '.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
        paths['png'] = folder / (name + '.png')
        preview = folder / '_preview'
        preview.mkdir(exist_ok=True)
        fig.savefig(preview / (name + '.png'), dpi=style['preview_dpi'], bbox_inches='tight', facecolor='white')
        paths['preview'] = preview / (name + '.png')
    finally:
        plt.close(fig)
    return paths


def plot_quality(study, output, style, cohort='Cold'):
    stats, coverage, detail = quality_data(study, cohort)
    fig, ax = plt.subplots(figsize=style['sizes']['quality'])
    generated = coverage.set_index('Tasks')['Generated']
    cert = stats[stats['Plot method'].eq('Certified')].set_index('Tasks').reindex(study['tasks'])
    # Disconnected hollow points explicitly mark conditional certified samples.
    full = cert['n'].eq(generated)
    _line(ax, cert.index, cert['mean'].where(full), 'Certified', style,
          label=f"HiGHS {study['benchmark']:g} s - certified mean", linewidth=2.)
    ax.fill_between(cert.index, (cert['mean'] - cert['sd']).where(full),
                    (cert['mean'] + cert['sd']).where(full), color=style['colors']['Certified'],
                    alpha=.20, linewidth=0)
    partial_label = False
    for task, row in cert[~full & cert['n'].gt(0)].iterrows():
        sd = row['sd'] if pd.notna(row['sd']) else 0.
        ax.errorbar(task, row['mean'], yerr=sd, fmt='o', markersize=8,
                    markerfacecolor='white', markeredgecolor=style['colors']['Certified'],
                    markeredgewidth=1.8, ecolor=style['colors']['Certified'], capsize=4,
                    label='HiGHS - certified subset' if not partial_label else '_nolegend_', zorder=5)
        ax.annotate(f"{int(row['n'])}/{int(generated.loc[task])}", (task, row['mean'] + sd),
                    xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8.7,
                    color=style['colors']['Certified'])
        partial_label = True
    for method in HEURISTICS:
        rows = stats[stats['Plot method'].eq(method)].set_index('Tasks').reindex(study['tasks'])
        _line(ax, rows.index, rows['mean'], method, style, linewidth=1.8, markersize=5.5)
        ax.fill_between(rows.index, (rows['mean'] - rows['sd']).clip(lower=0),
                        (rows['mean'] + rows['sd']).clip(upper=1),
                        color=style['colors'][method], alpha=.16, linewidth=0)
    ax.set(xlabel='Number of tasks', ylabel='Mean normalized accumulated benefit', xlim=(18, 62))
    ax.set_xticks(np.arange(20, 61, 5))
    if style['quality_ylim'] is not None:
        ax.set_ylim(style['quality_ylim'])
    elif not stats.empty:
        ax.set_ylim(max(0, float((stats['mean'] - stats['sd'].fillna(0)).min()) - .04),
                    min(1, float((stats['mean'] + stats['sd'].fillna(0)).max()) + .06))
    _axis(ax)
    ax.legend(loc='upper right', frameon=True, framealpha=.92)
    fig.tight_layout()
    paths = _save(fig, output, study['experiment'] + '1_solution_quality_certified_matched', style)
    return paths, stats, coverage


def plot_runtime(study, output, style):
    bench = study['warm'][study['warm']['Time Limit sec'].eq(study['benchmark'])].copy()
    bench['Plot method'] = bench['Start Method']
    heur = study['heuristics'].copy()
    heur['Plot method'] = heur['Start Method'].map({'GBStart': 'GreedyBenefit', 'GDStart': 'GreedyDensity'})
    bench = pd.concat([bench, heur], ignore_index=True)
    fig, ax = plt.subplots(figsize=style['sizes']['runtime'])
    for method in METHODS + HEURISTICS:
        group = bench[bench['Plot method'].eq(method)].groupby('Tasks')['End-to-end sec']
        med, q1, q3 = group.median(), group.quantile(.25), group.quantile(.75)
        _line(ax, med.index, med.values, method, style, linewidth=2.)
        ax.fill_between(med.index, q1, q3, color=style['colors'][method], alpha=.12, linewidth=0)
    ax.set(xlabel='Number of tasks', ylabel='End-to-end runtime (s, log scale)', yscale='log')
    ax.set_xticks(study['tasks'])
    ax.legend(frameon=True, ncol=2, loc='center left')
    _axis(ax)
    fig.tight_layout()
    return _save(fig, output, study['experiment'] + '2_end_to_end_computational_time', style)


def plot_effort(study, output, style, method='Cold'):
    rows = study['warm'][study['warm']['Time Limit sec'].eq(study['benchmark'])
                         & study['warm']['Start Method'].eq(method)]
    fig, axes = plt.subplots(1, 2, figsize=style['sizes']['effort'], constrained_layout=True)
    cmap, norm = plt.get_cmap('viridis'), plt.Normalize(min(study['tasks']), max(study['tasks']))
    rng = np.random.default_rng(7)
    for certified, marker, label in [(True, 'o', 'Certified optimum'), (False, 'X', 'Non-certified run')]:
        part = rows[rows['Optimal Certified'].eq(certified)]
        for ax, x in [(axes[0], part['Tasks'].to_numpy() + rng.uniform(-.7, .7, len(part))),
                      (axes[1], part['Variables'])]:
            ax.scatter(x, part['Solver Time sec'], c=part['Tasks'], cmap=cmap, norm=norm,
                        marker=marker, s=44 if certified else 64, alpha=.76,
                        edgecolors='black', linewidths=.4, label=label, zorder=3)
    group = rows.groupby('Tasks')['Solver Time sec']
    axes[0].plot(group.median().index, group.median(), color='black', marker='D',
                 linewidth=1.6, markersize=5, label='Median')
    axes[0].fill_between(group.median().index, group.quantile(.25), group.quantile(.75),
                         color='black', alpha=.10, linewidth=0, label='Interquartile range')
    for ax in axes:
        ax.axhline(study['benchmark'], color='#CC3311', linestyle='--', linewidth=1.5,
                   label=f"{study['benchmark']:g}-s configured limit")
        ax.set_yscale('log')
        ax.set_ylabel('Measured HiGHS solve time (s, log scale)')
        _axis(ax)
        ax.legend(fontsize=8, loc='lower right', frameon=True)
    axes[0].set(xlabel='Number of tasks', title='(a) HiGHS computational effort')
    axes[0].set_xticks(study['tasks'])
    axes[1].set(xscale='log', xlabel='Feasible state-transition arcs\n(binary decision variables)',
                title='(b) State-arc network size and runtime')
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.026, pad=.025)
    bar.set_label('Number of tasks')
    bar.set_ticks(study['tasks'])
    return _save(fig, output, study['experiment'] + '8_computational_effort_' + method, style)


def _plot_task_budget(study, table, column, ylabel, output, name, style, limits):
    fig, axes = plt.subplots(1, len(study['tasks']), figsize=style['sizes']['by_task'], sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.21, top=.78, wspace=.16)
    for ax, task in zip(np.atleast_1d(axes), study['tasks']):
        for method in METHODS:
            rows = table[table['Tasks'].eq(task) & table['Start Method'].eq(method)].sort_values('Time Limit sec')
            _line(ax, rows['Time Limit sec'], rows[column], method, style, markersize=4.5)
        _budget_axis(ax, study['budgets'])
        ax.set(title=f'{task} tasks', xlabel='Configured limit (s)', ylim=limits)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    _legend(fig, style)
    return _save(fig, output, name, style)


def _three_panel(study, table, time_column, definitions, output, name, style):
    fig, axes = plt.subplots(1, 3, figsize=style['sizes']['overview'])
    fig.subplots_adjust(left=.075, right=.99, bottom=.22, top=.78, wspace=.35)
    for ax, (column, label, limits) in zip(axes, definitions):
        for method in METHODS:
            rows = table[table['Start Method'].eq(method)].sort_values(time_column)
            _line(ax, rows[time_column], rows[column], method, style, markersize=5)
        _budget_axis(ax, study['budgets'])
        ax.set(ylabel=label, ylim=limits, xlabel='Elapsed solve wall time (s)' if 'Elapsed' in time_column else 'Configured limit (s)')
    _legend(fig, style)
    return _save(fig, output, name, style)


def plot_attribution(study, output, style):
    table = attribution_data(study)[0]
    fig, axes = plt.subplots(1, 2, figsize=style['sizes']['combined'])
    fig.subplots_adjust(left=.085, right=.99, bottom=.22, top=.79, wspace=.28)
    for ax, column, label in zip(axes, ['mean_greedy_gain', 'mean_injection_gain'],
            ['Mean gain over initial greedy (pp)', 'Mean gain over cold + greedy fallback (pp)']):
        for method in ('GBStart', 'GDStart'):
            rows = table[table['Start Method'].eq(method)].sort_values('Time Limit sec')
            _line(ax, rows['Time Limit sec'], 100 * rows[column], method, style)
        ax.axhline(0, color='#777777', linewidth=1.)
        _budget_axis(ax, study['budgets'])
        ax.set(xlabel='Configured limit (s)', ylabel=label)
    _legend(fig, style, methods=('GBStart', 'GDStart'))
    return _save(fig, output, study['experiment'] + '12_warm_start_attribution', style)


def replot_run(run_directory, experiment=None, output_directory=None, style=None,
               quality_cohort='Cold', include_budget_figures=True, require_full=True):
    """Append-only entry point for an already-running A/B notebook kernel."""
    style = style_config() if style is None else style
    study = load_run(run_directory, experiment, require_full=require_full)
    output = Path(output_directory) if output_directory else study['directory'] / 'paper_restyled'
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    with paper_style(style):
        figures['quality'], quality, coverage = plot_quality(study, output, style, quality_cohort)
        figures['runtime'] = plot_runtime(study, output, style)
        figures['effort'] = plot_effort(study, output, style)
        if include_budget_figures:
            figures['attribution'] = plot_attribution(study, output, style)
            table = budget_data(study, by_task=True)
            definitions = [('certified_pct', 'Optimality certified (%)', '3_certification_by_tasks_and_budget', (-3, 105)),
                ('solver_incumbents_pct', 'Solver-returned incumbent (%)', '9_budget_feasibility', (-3, 105)),
                ('mean_incumbent', 'Mean returned incumbent benefit', '10_budget_incumbent_value', (-.02, 1.02)),
                ('mean_gap_pct', 'Mean shortfall to known optimum (%)', '11_budget_optimality_shortfall', (-3, 105))]
            for col, label, suffix, lim in definitions:
                figures[suffix] = _plot_task_budget(study, table, col, label, output,
                    study['experiment'] + suffix, style, lim)
            overall = budget_data(study)
            figures['overview'] = _three_panel(study, overall, 'Time Limit sec', [
                ('solver_incumbents_pct', 'Returned incumbent (%)', (-3, 105)),
                ('mean_available', 'Mean available benefit (all instances)', (-.02, 1.02)),
                ('mean_available_gap_pct', 'Mean available shortfall (%)', (-3, 105))], output,
                study['experiment'] + '13_budget_quality_overview', style)
            clock = clock_summary(clock_data(study))
            figures['clock'] = _three_panel(study, clock, 'Elapsed solve wall sec', [
                ('available_plans_pct', 'Available plans (%)', (-3, 105)),
                ('mean_available', 'Mean available benefit (all instances)', (-.02, 1.02)),
                ('mean_available_gap_pct', 'Mean available shortfall (%)', (-3, 105))], output,
                study['experiment'] + '14_observed_anytime_quality', style)
    quality.to_csv(output / 'certified_matched_quality.csv', index=False)
    coverage.to_csv(output / 'certified_cohort_counts.csv', index=False)
    budget_data(study).to_csv(output / 'budget_status.csv', index=False)
    budget_data(study, True).to_csv(output / 'budget_by_task.csv', index=False)
    clock_summary(clock_data(study)).to_csv(output / 'clock_quality_overall.csv', index=False)
    attribution_data(study)[0].to_csv(output / 'injection_effect.csv', index=False)
    _write_single_tables(study, quality, coverage, output, quality_cohort)
    return {'directory': output, 'study': study, 'figures': figures,
            'quality': quality, 'coverage': coverage}


def _pm(mean, sd, digits):
    if pd.isna(mean):
        return '--'
    if pd.isna(sd):
        return f'${mean:.{digits}f}$'
    return f'${mean:.{digits}f} \\pm {sd:.{digits}f}$'


def _table_start(caption, label, headers, wide=False, sep=2.6):
    environment = 'table*' if wide else 'table'
    return [rf'\begin{{{environment}}}[!htbp]', r'\centering', r'\caption{' + caption + '}',
            r'\label{' + label + '}', r'\scriptsize', rf'\setlength{{\tabcolsep}}{{{sep}pt}}',
            r'\renewcommand{\arraystretch}{1.08}',
            r'\begin{tabular}{@{}' + 'c' * len(headers) + r'@{}}', r'\toprule',
            ' & '.join(headers) + r' \\', r'\midrule']


def _table_end(lines, note, wide=False):
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\begin{minipage}{0.98\textwidth}' if wide else r'\begin{minipage}{0.98\columnwidth}',
              r'\footnotesize', note, r'\end{minipage}', r'\end{table*}' if wide else r'\end{table}']
    return '\n'.join(lines) + '\n'


def _write_single_tables(study, quality, coverage, output, cohort):
    e = study['experiment']
    number = 1 if e == 'A' else 3
    lines = _table_start(f'Experiment {e}: solution quality on matched certified instances.',
        f'tab:exp{e}-quality', ['Tasks', r'\makecell{Normalized benefit\\Mean $\pm$ SD}',
        r'\makecell{Completed tasks\\Mean $\pm$ SD}', r'\makecell{Certified\\runs}',
        r'\makecell{Greedy gap\\benefit (\%)}'])
    for index, method in enumerate(('Certified',) + HEURISTICS):
        if index:
            lines.append(r'\midrule')
        lines += [r'\multicolumn{5}{@{}l}{\textit{Panel ' + chr(65 + index) + ': ' + LABELS[method] + r'}}\\',
                  r'\addlinespace[0.2em]']
        for task in study['tasks']:
            chosen = quality[quality['Tasks'].eq(task) & quality['Plot method'].eq(method)]
            n = int(coverage.loc[coverage['Tasks'].eq(task), 'Certified cohort'].iloc[0])
            cert = f'${n}/{len(study["seeds"])}$' if method == 'Certified' else '--'
            if chosen.empty:
                values = [str(task), '--', '--', cert, '--']
            else:
                r = chosen.iloc[0]
                values = [str(task), _pm(r['mean'], r['sd'], 4), _pm(r['tasks_mean'], r['tasks_sd'], 2),
                          cert, '--' if method == 'Certified' else f"${r['Mean-ratio shortfall pct']:.1f}$"]
            lines.append(' & '.join(values) + r' \\')
    note = (f'All three panels use the same certified-instance subset at each task count (selection: {cohort}). '
            f'The longest configured solver limit is {study["benchmark"]:g} s. Means and sample SD are conditional '
            'when certification is incomplete; they do not describe the unsolved instances. The last column is '
            'the relative shortfall between the displayed sample means. Certified counts use all generated '
            f'instances as denominator. Optimal status uses relative MIP tolerance {study["manifest"]["config"]["mip_rel_gap"]:g}.')
    (output / f'Table_{number}_Experiment_{e}_quality.tex').write_text(_table_end(lines, note), encoding='utf-8')
    headers = ['Tasks', r'\makecell{Runtime\\Mean $\pm$ SD}', r'\makecell{Maximum\\runtime}',
               r'\makecell{Variables\\mean}', r'\makecell{Constraints\\mean}']
    lines = _table_start(f'Experiment {e}: computational performance and generated state-arc MILP size.',
                         f'tab:exp{e}-runtime', headers, sep=2.4)
    bench = study['warm'][study['warm']['Time Limit sec'].eq(study['benchmark'])].copy()
    bench['Plot method'] = bench['Start Method']
    heur = study['heuristics'].copy()
    heur['Plot method'] = heur['Start Method'].map({'GBStart': 'GreedyBenefit', 'GDStart': 'GreedyDensity'})
    bench = pd.concat([bench, heur], ignore_index=True)
    for index, method in enumerate(METHODS + HEURISTICS):
        if index:
            lines.append(r'\midrule')
        unit, factor, digits = ('seconds', 1, 2) if method in METHODS else ('milliseconds', 1000, 3)
        lines += [r'\multicolumn{5}{@{}l}{\textit{Panel ' + chr(65 + index) + ': ' + LABELS[method] + ' (' + unit + r')}}\\', r'\addlinespace[0.2em]']
        for task, group in bench[bench['Plot method'].eq(method)].groupby('Tasks'):
            times = group['End-to-end sec'] * factor
            model = [f"${group[col].mean():,.0f}$".replace(',', '{,}') if method == 'Cold' else '--'
                     for col in ['Variables', 'Constraints']]
            lines.append(' & '.join([str(int(task)), _pm(times.mean(), times.std(), digits),
                f'${times.max():.{digits}f}$'] + model) + r' \\')
    note = ('All generated instances are included regardless of certification. End-to-end time includes '
            'network generation, matrix construction, native conversion, model loading and solving; warm '
            'variants also include greedy construction, mapping, validation and submission. The common MILP '
            'size is shown once. A configured solver limit is not a guaranteed elapsed-time deadline.')
    (output / f'Table_{number+1}_Experiment_{e}_runtime.tex').write_text(_table_end(lines, note), encoding='utf-8')


def _check_pair(a, b):
    if a['experiment'] != 'A' or b['experiment'] != 'B':
        raise ValueError('Expected Experiment A on the left and B on the right')
    for key in ['objective', 'threads', 'solver_seed', 'mip_rel_gap']:
        if a['manifest']['config'][key] != b['manifest']['config'][key]:
            raise ValueError('Incompatible experimental settings: ' + key)
    if a['tasks'] != b['tasks'] or a['seeds'] != b['seeds']:
        raise ValueError('Different task-size/seed cohorts')


def _clock_by_task(study):
    result = clock_data(study).groupby(['Tasks', 'Start Method', 'Elapsed solve wall sec'], as_index=False).agg(
        n=('Scenario', 'size'), certified=('Certified by time', 'sum'))
    result['certified_pct'] = 100 * result['certified'] / result['n']
    return result.rename(columns={'Elapsed solve wall sec': 'Time Limit sec'})


def _thresholds(studies, targets, observed=False):
    rows = []
    for study in studies:
        data = _clock_by_task(study) if observed else budget_data(study, True)
        for method in METHODS:
            for task in study['tasks']:
                group = data[data['Start Method'].eq(method) & data['Tasks'].eq(task)].sort_values('Time Limit sec')
                for target in targets:
                    hit = group[group['certified_pct'].ge(target)]
                    rows.append({'Experiment': study['experiment'], 'Tasks': task, 'Start Method': method,
                        'Target pct': target, 'First tested limit sec': np.nan if hit.empty else hit['Time Limit sec'].iloc[0],
                        'Reached': not hit.empty, 'Maximum tested sec': study['benchmark']})
    return pd.DataFrame(rows)


def combined_figures(a, b, output_directory, style=None, thresholds=(80, 95),
                     frontier_methods=METHODS, include_attribution=True):
    """Two central A14/B14 panels, integrated budgets, thresholds and LaTeX tables."""
    _check_pair(a, b)
    if not thresholds or any(not 0 < float(t) <= 100 for t in thresholds):
        raise ValueError('Thresholds must be percentages in (0, 100]')
    if any(method not in METHODS for method in frontier_methods):
        raise ValueError('Unknown frontier method')
    style = style_config() if style is None else style
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    studies = (a, b)
    figures = {}
    clocks = {s['experiment']: clock_summary(clock_data(s)) for s in studies}
    with paper_style(style):
        # Exactly the middle-panel statistic of A14 and B14, on a shared scale.
        fig, axes = plt.subplots(1, 2, figsize=style['sizes']['combined'], sharey=True)
        fig.subplots_adjust(left=.075, right=.99, bottom=.22, top=.79, wspace=.17)
        for ax, study in zip(axes, studies):
            data = clocks[study['experiment']]
            for method in METHODS:
                rows = data[data['Start Method'].eq(method)].sort_values('Elapsed solve wall sec')
                _line(ax, rows['Elapsed solve wall sec'], rows['mean_available'], method, style)
            _budget_axis(ax, study['budgets'])
            ax.set(xlabel='Elapsed solve wall time (s)', ylim=style['available_ylim'],
                   title=f"({'a' if study['experiment']=='A' else 'b'}) Experiment {study['experiment']} ({'4 × 4' if study['experiment']=='A' else '6 × 7'})")
        axes[0].set_ylabel('Mean available normalized benefit')
        _legend(fig, style)
        figures['warm_effect'] = _save(fig, output, 'AB14_observed_available_benefit', style)
        # Both clocks are exported explicitly. Observed time is the appropriate
        # choice for a strict solve-time budget; configured-limit runs can overrun.
        for observed, method in [(clock, m) for clock in (False, True) for m in frontier_methods]:
            fig, axes = plt.subplots(1, len(a['tasks']), figsize=style['sizes']['by_task'], sharey=True)
            fig.subplots_adjust(left=.065, right=.99, bottom=.21, top=.76, wspace=.16)
            for ax, task in zip(np.atleast_1d(axes), a['tasks']):
                for study in studies:
                    part = _clock_by_task(study) if observed else budget_data(study, True)
                    rows = part[part['Start Method'].eq(method) & part['Tasks'].eq(task)]
                    ax.plot(rows['Time Limit sec'], rows['certified_pct'],
                            color=style['colors'][study['experiment']], marker='o' if study is a else 's',
                            linestyle='-' if study is a else '--', markerfacecolor=style['colors']['A'] if study is a else 'white',
                            label='Experiment ' + study['experiment'])
                for index, target in enumerate(thresholds):
                    ax.axhline(target, color='#707070', linestyle=':' if index == 0 else '-.',
                               linewidth=.85, alpha=.7)
                ax.axvspan(600, 1800, color=style['colors']['B'], alpha=.065, zorder=0)
                ax.axvline(600, color='#888888', linewidth=.7, linestyle=':')
                _budget_axis(ax, b['budgets'])
                ax.set(title=f'{task} tasks', xlabel='Elapsed solve time (s)' if observed else 'Configured limit (s)', ylim=(-3, 105))
                ax.set_yticks([0, 25, 50, 75, 100])
            np.atleast_1d(axes)[0].set_ylabel('Optimality certified (%)')
            handles = np.atleast_1d(axes)[0].get_legend_handles_labels()[0]
            handles += [Line2D([], [], color='#707070', linestyle=':' if i == 0 else '-.',
                               label=f'{target:g}% target') for i, target in enumerate(thresholds)]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99), ncol=4, frameon=False)
            fig.text(.5, .84, LABELS[method], ha='center', fontsize=11)
            prefix = 'AB5_observed_certification_' if observed else 'AB5_budget_certification_'
            figures[('observed_frontier_' if observed else 'frontier_') + method] = _save(fig, output, prefix + method, style)
        # Budget dashboard uses separate runs; it must not be labelled as a trajectory.
        fig, axes = plt.subplots(2, 3, figsize=style['sizes']['dashboard'])
        fig.subplots_adjust(left=.075, right=.99, bottom=.13, top=.87, hspace=.56, wspace=.30)
        definitions = [('solver_incumbents_pct', 'Returned incumbent (%)', (-3, 105)),
                       ('mean_available', 'Mean available benefit', (-.02, 1.02)),
                       ('mean_available_gap_pct', 'Available shortfall to optimum (%)', (-3, 105))]
        for row, study in enumerate(studies):
            table = budget_data(study)
            for ax, (col, label, lim) in zip(axes[row], definitions):
                for method in METHODS:
                    data = table[table['Start Method'].eq(method)].sort_values('Time Limit sec')
                    _line(ax, data['Time Limit sec'], data[col], method, style, markersize=5)
                _budget_axis(ax, study['budgets'])
                ax.set(xlabel='Configured limit (s)', ylabel=label, ylim=lim)
                ax.set_title(f"Experiment {study['experiment']} · n={study['n']}" if col != 'mean_available_gap_pct'
                             else f"Experiment {study['experiment']} · known optima={len(study['references'])}")
        _legend(fig, style)
        figures['dashboard'] = _save(fig, output, 'AB_budget_quality_dashboard', style)
        if include_attribution:
            fig, axes = plt.subplots(1, 2, figsize=style['sizes']['combined'], sharey=True)
            fig.subplots_adjust(left=.085, right=.99, bottom=.22, top=.79, wspace=.18)
            bound = max(.05, max(100 * attribution_data(s)[0]['mean_injection_gain'].abs().max() for s in studies) * 1.25)
            for ax, study in zip(axes, studies):
                table = attribution_data(study)[0]
                for method in ('GBStart', 'GDStart'):
                    rows = table[table['Start Method'].eq(method)].sort_values('Time Limit sec')
                    _line(ax, rows['Time Limit sec'], 100 * rows['mean_injection_gain'], method, style)
                ax.axhline(0, color='#777777', linewidth=1.)
                _budget_axis(ax, study['budgets'])
                ax.set(xlabel='Configured limit (s)', title='Experiment ' + study['experiment'], ylim=(-bound, bound))
            axes[0].set_ylabel('Gain over cold + same greedy fallback (pp)')
            _legend(fig, style, methods=('GBStart', 'GDStart'))
            figures['injection'] = _save(fig, output, 'AB15_paired_injection_effect', style)
    limits = _thresholds(studies, thresholds)
    limits.to_csv(output / 'AB_certification_thresholds.csv', index=False)
    observed_limits = _thresholds(studies, thresholds, observed=True)
    observed_limits.to_csv(output / 'AB_observed_certification_thresholds.csv', index=False)
    for study in studies:
        e = study['experiment']
        clocks[e].to_csv(output / f'{e}_observed_clock.csv', index=False)
        clock_data(study).to_csv(output / f'{e}_observed_clock_detail.csv', index=False)
    status = pd.concat([budget_data(s).assign(Experiment=s['experiment']) for s in studies], ignore_index=True)
    status.to_csv(output / 'AB_budget_status.csv', index=False)
    attribution = pd.concat([attribution_data(s)[0].assign(Experiment=s['experiment']) for s in studies], ignore_index=True)
    attribution.to_csv(output / 'AB_paired_injection_effect.csv', index=False)
    overruns = pd.concat([s['warm'].assign(Experiment=s['experiment']) for s in studies], ignore_index=True)
    overruns = overruns[['Experiment'] + KEYS + ['Start Method', 'Time Limit sec', 'Solver Time sec',
        'HiGHS Internal Time sec', 'Overrun sec', 'Optimal Certified', 'Solver Has Incumbent']]
    overruns.to_csv(output / 'AB_measured_time_audit.csv', index=False)
    _write_combined_tables(studies, status, limits, output)
    _write_clock_table(clocks, output)
    _write_captions(output)
    metadata = {'data_revision': DATA_REVISION, 'objective': a['manifest']['config']['objective'],
        'cohorts': {s['experiment']: s['n'] for s in studies}, 'thresholds_pct': list(thresholds),
        'clock_sources_sec': {'A': a['benchmark'], 'B': b['benchmark']},
        'clock': 'callbacks at or before the point; validated initial plan at zero; idle=0 otherwise',
        'configured_limits': 'independent restarts, not strict measured deadlines',
        'selection': 'All cases for budget and clock plots; certification filter only for main quality',
        'figure_style': style}
    (output / 'publication_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return {'directory': output, 'figures': figures, 'budget_status': status,
            'thresholds': limits, 'observed_thresholds': observed_limits,
            'attribution': attribution, 'clocks': clocks}


def _write_combined_tables(studies, status, limits, output):
    headers = ['Exp.', r'\makecell{Limit\\(s)}', 'Start', r'\makecell{Certified\\$n$}',
        r'\makecell{Feasible\\uncertified $n$}', r'\makecell{No solver\\incumbent $n$}',
        r'\makecell{Mean incumbent\\($n$)}', r'\makecell{Mean available\\benefit}',
        r'\makecell{Mean shortfall\\(\%); $n$}']
    lines = _table_start('Configured-limit results for Experiments A and B.', 'tab:AB-budget-status', headers, wide=True, sep=3.1)
    for e in ['A', 'B']:
        if e == 'B':
            lines.append(r'\midrule')
        group = status[status['Experiment'].eq(e)].sort_values(['Time Limit sec', 'Start Method'])
        for _, r in group.iterrows():
            incumbent = '-- (0)' if pd.isna(r['mean_incumbent']) else f"{r['mean_incumbent']:.4f} ({int(r['solver_incumbents'])})"
            gap = '--; 0' if pd.isna(r['mean_gap_pct']) else f"{r['mean_gap_pct']:.2f}; {int(r['gap_n'])}"
            lines.append(' & '.join([e, f"{r['Time Limit sec']:g}",
                {'Cold': 'Cold', 'GBStart': 'GB', 'GDStart': 'GD'}[r['Start Method']], str(int(r['certified'])),
                str(int(r['feasible_uncertified'])), str(int(r['no_incumbent'])), incumbent,
                f"{r['mean_available']:.4f}", gap]) + r' \\')
    note = ('Each experiment has 150 instances. Certified, feasible uncertified and no solver incumbent '
        'are disjoint categories summing to 150. The mean incumbent excludes missing solver solutions; '
        'its sample count is shown. Available benefit averages over all 150 instances and assigns idle '
        'value zero separately when no solver plan is returned. Shortfalls are computed per instance '
        'against a certified optimum of the same model, then averaged only over returned incumbents '
        'with a known optimum. There are 150 known references in A and 116 in B. GB and GD denote '
        'Benefit and Density starts. Limits are configured HiGHS allowances; measured times may exceed them. '
        'These are independent runs, not observations at exact wall-clock deadlines.')
    (output / 'Table_AB_budget_status.tex').write_text(_table_end(lines, note, True), encoding='utf-8')
    headers = ['Tasks', r'Target (\%)', r'\makecell{Cold\\A / B}',
               r'\makecell{Benefit start\\A / B}', r'\makecell{Density start\\A / B}']
    lines = _table_start('First tested configured limit reaching a certification target (seconds).',
                         'tab:AB-certification-thresholds', headers)
    for task in studies[0]['tasks']:
        for target in sorted(limits['Target pct'].unique()):
            values = [str(task), f'{target:g}']
            for method in METHODS:
                pair = []
                for e in ['A', 'B']:
                    match = limits[limits['Experiment'].eq(e) & limits['Tasks'].eq(task)
                                   & limits['Start Method'].eq(method) & limits['Target pct'].eq(target)].iloc[0]
                    pair.append(f"{match['First tested limit sec']:g}" if match['Reached'] else '--')
                values.append(' / '.join(pair))
            lines.append(' & '.join(values) + r' \\')
    note = ('A / B gives the respective experiments. A dash means the target was not reached at any '
        'tested limit (maximum 600 s in A and 1800 s in B). Targets are descriptive certification '
        'thresholds, not humanitarian service standards. No interpolation or extrapolation is used. '
        'Each task size has 30 instances. Configured limits can differ from measured elapsed times.')
    (output / 'Table_AB_certification_thresholds.tex').write_text(_table_end(lines, note), encoding='utf-8')


def _write_captions(output):
    text = r'''% Needs graphicx, booktabs, makecell. Copy only the figures needed in the paper.
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{AB14_observed_available_benefit.pdf}
\caption{Available solution quality during a single long search: (a) Experiment A,
(b) Experiment B. Each curve averages all 150 instances. Validated greedy starts
are available at solve time zero; subsequent values use only incumbents observed
by the stated elapsed wall time. Cold searches without an observed incumbent
contribute idle value zero to this availability metric. Preparation is excluded.
The source search limits are 600 s in A and 1800 s in B. Lines connect the recorded
checkpoints and do not identify exact improvement times. These curves show
solution availability; they alone do not isolate the effect of injecting a start.}
\label{fig:AB-observed-quality}
\end{figure*}

% Recommended certification frontier when discussing actual solve-time budgets.
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{AB5_observed_certification_Cold.pdf}
\caption{Observed certification within a single long cold search per instance.
Each point counts the instances whose certified search had finished by the
stated elapsed solve time, out of 30 at that task count. Horizontal lines mark
80\% and 95\% targets. The shaded extension is tested only in Experiment B.
Preparation time is excluded. These checkpoints respect the recorded solve
clock and differ from the final outcomes of independent configured-limit runs.}
\label{fig:AB-observed-certification}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{AB5_budget_certification_Cold.pdf}
\caption{Certification under configured solver limits, by task count, for cold
HiGHS searches in Experiments A and B. Each point summarizes 30 independent
instances. Horizontal lines mark the descriptive 80\% and 95\% targets. The
shaded region contains the 600--1800-s extension tested only in B. Configured
limits are not strict elapsed-time deadlines; actual times are reported separately.}
\label{fig:AB-certification-budget}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{AB_budget_quality_dashboard.pdf}
\caption{Independent-run outcomes by configured limit. Rows correspond to
Experiments A and B. Returned-incumbent rates and mean available benefit use
all 150 instances. Available benefit assigns idle value zero separately when
no plan is returned; a validated start is available in each warm treatment.
The right panels average available-solution shortfalls over known certified
references only (150 in A, 116 in B), including idle values. These are different
statistics from conditional mean solver incumbents in Table~\ref{tab:AB-budget-status}.
Model and start preparation are excluded from the configured solver limit;
actual solve times may overrun that limit.}
\label{fig:AB-budget-quality}
\end{figure*}

% Optional supplementary attribution figure; use with the paired CSV counts.
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{AB15_paired_injection_effect.pdf}
\caption{Paired mean incremental value of injecting a start rather than keeping
the same greedy as an external fallback. For each instance and configured limit,
the difference is $z_{\mathrm{warm}}-\max\{z_{\mathrm{cold}},z_{\mathrm{greedy}}\}$,
in percentage points of normalized benefit. Negative differences are retained.
All 150 instances are included in each experiment. Preparation and elapsed-time
overruns are reported separately; these are paired configured-limit comparisons.}
\label{fig:AB-injection-effect}
\end{figure*}
'''
    (output / 'Figures_AB_overleaf.tex').write_text(text, encoding='utf-8')


def _write_clock_table(clocks, output):
    headers = ['Exp.', r'\makecell{Elapsed\\(s)}', 'Start', r'\makecell{Certified\\$n$}',
        r'\makecell{Observed feasible\\uncertified $n$}', r'\makecell{No observed solver\\incumbent $n$}',
        r'\makecell{Mean observed\\incumbent ($n$)}', r'\makecell{Mean available\\benefit}',
        r'\makecell{Available shortfall\\(\%); $n$}']
    lines = _table_start('Observed solution availability at elapsed solve-time checkpoints.',
        'tab:AB-observed-budget', headers, wide=True, sep=2.8)
    for e in ['A', 'B']:
        if e == 'B':
            lines.append(r'\midrule')
        table = clocks[e].sort_values(['Elapsed solve wall sec', 'Start Method'])
        for _, r in table.iterrows():
            count = int(r['solver_incumbents'])
            mean = '-- (0)' if not count else f"{r['mean_observed_incumbent']:.4f} ({count})"
            shortfall = '--; 0' if not r['known_references'] else f"{r['mean_available_gap_pct']:.2f}; {int(r['known_references'])}"
            lines.append(' & '.join([e, f"{r['Elapsed solve wall sec']:g}",
                {'Cold': 'Cold', 'GBStart': 'GB', 'GDStart': 'GD'}[r['Start Method']], str(int(r['certified'])),
                str(count - int(r['certified'])), str(int(r['n']) - count), mean,
                f"{r['mean_available']:.4f}", shortfall]) + r' \\')
    note = ('All checkpoints use one long search per instance and initialization (600-s source limit in A, '
        '1800 s in B). Only callbacks observed by the checkpoint are used. Certified, observed feasible '
        'uncertified and no observed solver incumbent partition the 150 instances; these categories '
        'describe observations and do not imply infeasibility when an incumbent has not been observed. '
        'A validated greedy start is available separately at time zero in both warm treatments. '
        'The available-benefit mean includes all 150 cases and idle value zero if necessary. '
        'Shortfalls include available starts or idle plans and use only the 150 known optima in A '
        'and 116 in B. Missing solver incumbents are excluded only from their conditional mean. '
        'Preparation time is excluded.')
    (output / 'Table_AB_observed_budget.tex').write_text(_table_end(lines, note, True), encoding='utf-8')
