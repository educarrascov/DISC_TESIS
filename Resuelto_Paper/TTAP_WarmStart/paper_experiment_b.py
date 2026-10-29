"""Six-budget B protocol, sharing A's unchanged generator and native MILP runner."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import pandas as pd

from objective_profiles import ObjectiveSpec
from study_runner import StudyConfig, TREATMENTS, atomic_csv, run_study

B_PAPER_PROTOCOL = 'TTAP-B-paper-six-budgets-v1'
B_PAPER_BUDGETS = (1., 5., 15., 60., 600., 1800.)
B_PAPER_TASKS = (20, 30, 40, 50, 60)
B_BENCHMARK_BUDGET = 1800.


@dataclass(frozen=True)
class PaperBConfig(StudyConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.experiment != 'B' or self.budgets != B_PAPER_BUDGETS:
            raise ValueError('Paper B requires B and budgets 1/5/15/60/600/1800 s')
        if self.reference_budgets:
            raise ValueError('All long budgets are included; do not run duplicate references')

    def as_dict(self):
        result = super().as_dict()
        result['paper_protocol'] = B_PAPER_PROTOCOL
        result['paper_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    @property
    def full_cohort(self):
        return (set(self.tasks) == set(B_PAPER_TASKS)
                and set(self.seeds) == set(range(1, 31))
                and self.budgets == B_PAPER_BUDGETS)

    def workload(self):
        result = super().workload()
        cases = len(self.tasks) * len(self.seeds)
        result.update(benchmark_budget_sec=B_BENCHMARK_BUDGET,
                      heuristic_runs=2 * cases, expected_metric_rows=20 * cases,
                      paper_protocol=B_PAPER_PROTOCOL)
        return result


def paper_b_configuration(mode='pilot', objective=None):
    if mode == 'pilot':
        tasks, seeds = (20,), (1,)
    elif mode == 'screen':
        tasks, seeds = (20, 40, 60), (1, 2, 3)
    elif mode == 'full':
        tasks, seeds = B_PAPER_TASKS, tuple(range(1, 31))
    else:
        raise ValueError('MODE must be pilot, screen or full')
    return PaperBConfig('B', tasks, seeds, B_PAPER_BUDGETS, (), objective or ObjectiveSpec())


def validate_paper_b_grid(raw, manifest, allow_pilot=False):
    cfg = manifest['config']
    if cfg.get('paper_protocol') != B_PAPER_PROTOCOL or cfg['experiment'] != 'B':
        raise ValueError('This is not the six-budget Experiment B paper protocol')
    if tuple(cfg['budgets']) != B_PAPER_BUDGETS or cfg['reference_budgets']:
        raise ValueError('All three treatments need all six budgets; no duplicate reference phase')
    if (not cfg['tasks'] or not cfg['seeds'] or set(cfg['tasks']) - set(B_PAPER_TASKS)
            or len(set(cfg['tasks'])) != len(cfg['tasks'])
            or len(set(cfg['seeds'])) != len(cfg['seeds'])
            or any(s not in range(1, 31) for s in cfg['seeds'])):
        raise ValueError('Invalid paper task sizes or seeds')
    full = set(cfg['tasks']) == set(B_PAPER_TASKS) and set(cfg['seeds']) == set(range(1, 31))
    if not full and not allow_pilot:
        raise ValueError('Paper export requires all five sizes and 30 seeds; use allow_pilot for a preview')
    if raw.empty or not raw['Configuration'].eq('B').all():
        raise ValueError('Missing or mixed experiment configurations')
    if not raw['Study Stage'].isin(['Warm', 'Heuristic']).all():
        raise ValueError('Unexpected reference or other rows')
    warm = raw[raw['Study Stage'].eq('Warm')]
    keys = ['Tasks', 'Seed', 'Time Limit sec', 'Start Method']
    expected = {(t, s, b, m) for t in cfg['tasks'] for s in cfg['seeds']
                for b in B_PAPER_BUDGETS for m in TREATMENTS}
    if warm.duplicated(keys).any() or set(warm[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Incomplete paper B grid: each task/seed needs all 18 solver jobs')
    heur, hkeys = raw[raw['Study Stage'].eq('Heuristic')], ['Tasks', 'Seed', 'Start Method']
    expected_h = {(t, s, m) for t in cfg['tasks'] for s in cfg['seeds'] for m in ('GBStart', 'GDStart')}
    if heur.duplicated(hkeys).any() or set(heur[hkeys].itertuples(index=False, name=None)) != expected_h:
        raise ValueError('Missing or duplicate matched greedy baselines')
    if raw.groupby(['Tasks', 'Seed'])['Scenario'].nunique().ne(1).any():
        raise ValueError('A task/seed pair identifies multiple scenarios')
    return full


def run_paper_experiment_b(config, output, resume=True):
    if not isinstance(config, PaperBConfig):
        raise TypeError('Use paper_b_configuration to construct this experiment')
    output, started = Path(output), datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    result = run_study(config, output, phase='warm', resume=resume)
    session = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
               'session_wall_sec': time.perf_counter() - clock,
               'metrics_present': len(result), 'config_sha256': config.fingerprint}
    path = output / 'batch_sessions.csv'
    sessions = pd.read_csv(path) if path.exists() else pd.DataFrame()
    atomic_csv(pd.concat([sessions, pd.DataFrame([session])], ignore_index=True), path)
    return result
