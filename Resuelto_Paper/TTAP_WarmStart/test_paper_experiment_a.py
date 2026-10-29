"""Paper-cohort completeness, honest missing data and publication export checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from objective_profiles import ObjectiveSpec
from paper_experiment_a import (PAPER_BUDGETS, PaperAConfig, paper_configuration,
                               validate_paper_grid)
from paper_a_report import prepare_paper_data, latex_tables, section_fragment
from study_runner import configuration, manifest_for


def fixture(mode='pilot'):
    config = paper_configuration(mode)
    spec = config.objective
    rows = []
    for tasks in config.tasks:
        for seed in config.seeds:
            optimum = .75 - .003 * tasks + .0005 * seed
            common = {'Configuration': 'A', 'Scenario': f'A4x4_T{tasks}_S{seed}',
                      'Tasks': tasks, 'Seed': seed, 'Objective SHA256': spec.fingerprint,
                      'Model SHA256': f'model-{tasks}-{seed}',
                      'Scenario SHA256': f'scenario-{tasks}-{seed}', 'HiGHS Version': '1.11.0',
                      'Threads': 1, 'Random Seed': 0}
            for method, factor in [('GBStart', .7), ('GDStart', .8)]:
                rows.append({**common, 'Study Stage': 'Heuristic', 'Start Method': method,
                             'Method': 'Greedy-'+method, 'Normalized Mission Value': factor*optimum,
                             'Physical Completed Tasks': int(tasks*.4), 'Has Incumbent': True,
                             'Optimal Certified': False, 'Runtime sec': .001,
                             'End-to-end sec': .001})
            for budget in PAPER_BUDGETS:
                for method in ('Cold', 'GBStart', 'GDStart'):
                    has = method != 'Cold' or budget >= 15
                    certified = budget >= 15
                    factor = {'Cold': .0, 'GBStart': .7, 'GDStart': .8}[method]
                    value = (optimum if certified else optimum*factor) if has else np.nan
                    rows.append({**common, 'Study Stage': 'Warm', 'Start Method': method,
                                 'Method': f'HiGHS-{budget:g}s-{method}', 'Time Limit sec': budget,
                                 'Normalized Mission Value': value, 'Solver Incumbent Value': value,
                                 'Has Incumbent': has, 'Solver Has Incumbent': has,
                                 'Optimal Certified': certified,
                                 'Warm Start Value': optimum*factor if method != 'Cold' else np.nan,
                                 'Physical Completed Tasks': int(tasks*.5) if has else np.nan,
                                 'Variables': tasks*1000+seed, 'Constraints': tasks*100+seed,
                                 'Solver Time sec': min(budget, 6.) + seed*.001,
                                 'End-to-end sec': min(budget, 6.)+.3+seed*.001})
    return pd.DataFrame(rows), manifest_for(config)


class PaperExperimentATests(unittest.TestCase):
    def test_workload_and_legacy_protocol_are_distinct(self):
        full = paper_configuration('full')
        self.assertTrue(full.full_cohort)
        self.assertEqual(full.workload()['warm_solver_runs'], 2250)
        self.assertEqual(full.workload()['heuristic_runs'], 300)
        self.assertEqual(full.workload()['expected_metric_rows'], 2550)
        self.assertEqual(full.workload()['reference_solver_runs'], 0)
        self.assertAlmostEqual(full.workload()['nominal_warm_solver_hours'], 85.125)
        self.assertEqual(configuration('A', 'full').budgets, (1., 5., 15., 60.))
        self.assertEqual(configuration('B', 'full').reference_budgets, (600., 1800.))
        self.assertEqual(json.loads(json.dumps(manifest_for(full))), manifest_for(full))

    def test_rejects_duplicate_reference_and_missing_benchmark_budget(self):
        with self.assertRaisesRegex(ValueError, 'duplicate references'):
            PaperAConfig('A', budgets=PAPER_BUDGETS, reference_budgets=(600.,))
        with self.assertRaisesRegex(ValueError, 'budgets'):
            PaperAConfig('A')

    def test_full_grid_includes_600_for_all_three_initializations(self):
        raw, manifest = fixture('full')
        self.assertTrue(validate_paper_grid(raw, manifest))
        mask = raw['Study Stage'].eq('Warm') & raw['Time Limit sec'].eq(600.) & raw['Start Method'].eq('GDStart')
        with self.assertRaisesRegex(ValueError, 'Incomplete paper grid'):
            validate_paper_grid(raw.drop(raw[mask].index[0]), manifest)
        with self.assertRaisesRegex(ValueError, 'Incomplete paper grid'):
            validate_paper_grid(pd.concat([raw, raw.iloc[[2]]], ignore_index=True), manifest)

    def test_pilot_does_not_emit_a_final_section(self):
        raw, manifest = fixture()
        with self.assertRaisesRegex(ValueError, 'all five sizes'):
            prepare_paper_data(raw, manifest)
        data = prepare_paper_data(raw, manifest, allow_pilot=True)
        self.assertFalse(data['full'])
        self.assertEqual(len(data['stats']), 5)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            latex_tables(data, p)
            section_fragment(data, p, True)
            self.assertFalse((p/'Section_4_1.tex').exists())
            self.assertTrue((p/'PILOT_ONLY.txt').is_file())
            text = (p/'Table_1_Experiment_A_quality.tex').read_text()
            self.assertIn('Pilot preview', text)
            self.assertIn('SD', text)
            self.assertNotIn('\\pm 0.0000', text)

    def test_full_statistics_have_five_variants_and_matched_denominators(self):
        raw, manifest = fixture('full')
        data = prepare_paper_data(raw, manifest)
        self.assertEqual(len(data['stats']), 25)
        self.assertTrue(data['stats']['n'].eq(30).all())
        self.assertEqual(len(data['references']), 150)
        greedy = data['stats'][data['stats']['Paper method'].eq('GreedyBenefit')]
        self.assertTrue(np.allclose(greedy['shortfall_mean_pct'], 30.))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            latex_tables(data, p)
            section_fragment(data, p, True)
            self.assertIn('Panel E', (p/'Table_1_Experiment_A_quality.tex').read_text())
            text = (p/'Section_4_1.tex').read_text()
            self.assertIn('150/150', text)
            self.assertNotIn('Pilot preview', text)

    def test_missing_incumbent_and_unknown_optimum_remain_distinct(self):
        raw, manifest = fixture('full')
        # One cold 600-s row has no incumbent, but a matched warm optimum exists.
        mask = raw['Tasks'].eq(20) & raw['Seed'].eq(1) & raw['Time Limit sec'].eq(600.) & raw['Start Method'].eq('Cold')
        raw.loc[mask, ['Has Incumbent', 'Solver Has Incumbent', 'Optimal Certified']] = False
        raw.loc[mask, ['Normalized Mission Value', 'Solver Incumbent Value', 'Physical Completed Tasks']] = np.nan
        # A different scenario has no optimality certificate at any budget.
        unknown = raw['Tasks'].eq(60) & raw['Seed'].eq(30) & raw['Study Stage'].eq('Warm')
        raw.loc[unknown, 'Optimal Certified'] = False
        data = prepare_paper_data(raw, manifest)
        cold = data['benchmark'][data['benchmark']['Paper method'].eq('Cold') & data['benchmark']['Tasks'].eq(20)
                                 & data['benchmark']['Seed'].eq(1)].iloc[0]
        self.assertTrue(pd.isna(cold['Normalized Mission Value']))
        self.assertEqual(cold['Available value'], 0.)
        self.assertAlmostEqual(cold['Benchmark shortfall pct'], 100.)
        self.assertEqual(len(data['references']), 149)
        rows = data['stats'][data['stats']['Tasks'].eq(60)]
        self.assertTrue(rows['n'].eq(30).all())
        self.assertTrue(rows['n_known_optima'].eq(29).all())


if __name__ == '__main__':
    unittest.main()
