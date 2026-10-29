"""Experiment A paper protocol: five budgets and three native HiGHS starts.

The legacy A/B runner is reused without changing its generator or MILP.
This profile has its own manifest and output directory; old four-budget runs
are never silently appended to the paper cohort.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import pandas as pd

from objective_profiles import ObjectiveSpec
from study_runner import StudyConfig, TREATMENTS, atomic_csv, run_study

PAPER_PROTOCOL = 'TTAP-A-paper-five-budgets-v1'
PAPER_BUDGETS = (1., 5., 15., 60., 600.)
PAPER_TASKS = (20, 30, 40, 50, 60)
BENCHMARK_BUDGET = 600.


@dataclass(frozen=True)
class PaperAConfig(StudyConfig):
    def __post_init__(self):
        super().__post_init__()
        if self.experiment != 'A' or self.budgets != PAPER_BUDGETS:
            raise ValueError('The paper A profile requires A and budgets 1/5/15/60/600 s')
        if self.reference_budgets:
            raise ValueError('Cold at 600 s is already included; do not run duplicate references')

    def as_dict(self):
        result = super().as_dict()
        result['paper_protocol'] = PAPER_PROTOCOL
        result['paper_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    @property
    def full_cohort(self):
        return (set(self.tasks) == set(PAPER_TASKS)
                and set(self.seeds) == set(range(1, 31))
                and self.budgets == PAPER_BUDGETS)

    def workload(self):
        result = super().workload()
        result.update(benchmark_budget_sec=BENCHMARK_BUDGET,
                      heuristic_runs=2 * len(self.tasks) * len(self.seeds),
                      expected_metric_rows=len(self.tasks) * len(self.seeds)
                                           * (2 + len(TREATMENTS) * len(self.budgets)),
                      paper_protocol=PAPER_PROTOCOL)
        return result


def paper_configuration(mode='full', objective=None):
    if mode == 'pilot':
        tasks, seeds = (20,), (1,)
    elif mode == 'screen':
        tasks, seeds = (20, 40, 60), (1, 2, 3)
    elif mode == 'full':
        tasks, seeds = PAPER_TASKS, tuple(range(1, 31))
    else:
        raise ValueError('MODE must be pilot, screen or full')
    return PaperAConfig('A', tasks, seeds, PAPER_BUDGETS, (), objective or ObjectiveSpec())


def validate_paper_grid(raw, manifest, allow_pilot=False):
    cfg = manifest['config']
    if cfg.get('paper_protocol') != PAPER_PROTOCOL or cfg['experiment'] != 'A':
        raise ValueError('This is not the five-budget Experiment A paper protocol')
    if tuple(cfg['budgets']) != PAPER_BUDGETS or cfg['reference_budgets']:
        raise ValueError('All three treatments must include 600 s; no duplicate reference phase')
    if (set(cfg['tasks']) - set(PAPER_TASKS) or not cfg['tasks'] or not cfg['seeds']
            or len(set(cfg['tasks'])) != len(cfg['tasks'])
            or len(set(cfg['seeds'])) != len(cfg['seeds'])
            or any(s not in range(1, 31) for s in cfg['seeds'])):
        raise ValueError('Invalid paper task sizes or seeds')
    is_full = (set(cfg['tasks']) == set(PAPER_TASKS)
               and set(cfg['seeds']) == set(range(1, 31)))
    if not is_full and not allow_pilot:
        raise ValueError('Paper export requires all five sizes and 30 seeds; use allow_pilot for a preview')
    if raw.empty or not raw['Configuration'].eq('A').all():
        raise ValueError('Missing or mixed experiment configurations')
    if not raw['Study Stage'].isin(['Warm', 'Heuristic']).all():
        raise ValueError('Unexpected reference or other rows in the paper cohort')
    warm = raw[raw['Study Stage'].eq('Warm')]
    keys = ['Tasks', 'Seed', 'Time Limit sec', 'Start Method']
    expected = {(t, s, b, m) for t in cfg['tasks'] for s in cfg['seeds']
                for b in PAPER_BUDGETS for m in TREATMENTS}
    actual = set(warm[keys].itertuples(index=False, name=None))
    if warm.duplicated(keys).any() or actual != expected:
        raise ValueError('Incomplete paper grid: every task/seed needs all 15 solver jobs, including 600 s')
    heur = raw[raw['Study Stage'].eq('Heuristic')]
    expected_h = {(t, s, m) for t in cfg['tasks'] for s in cfg['seeds']
                  for m in ('GBStart', 'GDStart')}
    hkeys = ['Tasks', 'Seed', 'Start Method']
    if heur.duplicated(hkeys).any() or set(heur[hkeys].itertuples(index=False, name=None)) != expected_h:
        raise ValueError('Missing or duplicate matched greedy baselines')
    if raw.groupby(['Tasks', 'Seed'])['Scenario'].nunique().ne(1).any():
        raise ValueError('A task/seed pair identifies multiple scenarios')
    return is_full


def run_paper_experiment_a(config, output, resume=True):
    if not isinstance(config, PaperAConfig):
        raise TypeError('Use paper_configuration to construct this experiment')
    output = Path(output)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    result = run_study(config, output, phase='warm', resume=resume)
    session = {'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
               'session_wall_sec': time.perf_counter() - clock,
               'metrics_present': len(result), 'config_sha256': config.fingerprint}
    path = output / 'batch_sessions.csv'
    sessions = pd.read_csv(path) if path.exists() else pd.DataFrame()
    atomic_csv(pd.concat([sessions, pd.DataFrame([session])], ignore_index=True), path)
    return result
