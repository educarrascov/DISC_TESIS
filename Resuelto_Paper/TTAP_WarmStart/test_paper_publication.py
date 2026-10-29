"""Shared A/B publication semantics and the complete B grid."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper_experiment_b import (B_PAPER_BUDGETS, PaperBConfig, paper_b_configuration,
                                validate_paper_b_grid)
from paper_a_report import (attribution_counts, latex_tables, prepare_paper_data,
                            section_fragment, _save, PUBLICATION_STYLE, budget_figures)
from study_runner import manifest_for
from test_paper_experiment_a import fixture


def b_fixture(mode='full'):
    raw, _ = fixture(mode)
    longest = raw[raw['Time Limit sec'].eq(600.)].copy()
    longest['Time Limit sec'] = 1800.
    longest['Method'] = longest['Method'].str.replace('600s', '1800s', regex=False)
    raw = pd.concat([raw, longest], ignore_index=True)
    raw['Configuration'] = 'B'
    raw['Scenario'] = raw['Scenario'].str.replace('A4x4', 'B6x7', regex=False)
    return raw, manifest_for(paper_b_configuration(mode))


class PublicationTests(unittest.TestCase):
    def test_pilot_budget_figures_preserve_empty_unrun_sizes(self):
        import matplotlib.pyplot as plt
        raw, manifest = b_fixture('pilot')
        data = prepare_paper_data(raw, manifest, True, experiment='B')
        with tempfile.TemporaryDirectory() as directory, plt.rc_context(PUBLICATION_STYLE):
            quality, _ = budget_figures(data, Path(directory), 72)
            self.assertEqual(set(quality['Tasks']), {20})
            self.assertTrue(quality['n'].eq(1).all())
            self.assertEqual(len(list(Path(directory).glob('B*.pdf'))), 5)

    def test_b_workload_and_no_duplicate_references(self):
        config = paper_b_configuration('full')
        self.assertTrue(config.full_cohort)
        self.assertEqual(config.budgets, (1., 5., 15., 60., 600., 1800.))
        self.assertEqual(config.workload()['warm_solver_runs'], 2700)
        self.assertEqual(config.workload()['expected_metric_rows'], 3000)
        self.assertEqual(config.workload()['reference_solver_runs'], 0)
        self.assertAlmostEqual(config.workload()['nominal_warm_solver_hours'], 310.125)
        self.assertEqual(json.loads(json.dumps(manifest_for(config))), manifest_for(config))
        with self.assertRaises(ValueError):
            PaperBConfig('B', budgets=B_PAPER_BUDGETS, reference_budgets=(1800.,))

    def test_b_exact_grid_and_shared_benchmark_statistics(self):
        raw, manifest = b_fixture()
        self.assertTrue(validate_paper_b_grid(raw, manifest))
        data = prepare_paper_data(raw, manifest, experiment='B')
        self.assertEqual(data['benchmark_budget'], 1800.)
        self.assertEqual(len(data['warm']), 2700)
        self.assertEqual(len(data['stats']), 25)
        self.assertTrue(data['stats']['n'].eq(30).all())
        with self.assertRaisesRegex(ValueError, 'Incomplete paper B grid'):
            validate_paper_b_grid(raw.iloc[:-1], manifest)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            latex_tables(data, output)
            section_fragment(data, output, True)
            self.assertTrue((output / 'Table_3_Experiment_B_quality.tex').is_file())
            self.assertTrue((output / 'Table_4_Experiment_B_runtime.tex').is_file())
            self.assertIn('1800-s', (output / 'Section_4_2.tex').read_text())
            self.assertFalse((output / 'Section_4_1.tex').exists())

    def test_b_unknown_references_do_not_become_best_known_optima(self):
        raw, manifest = b_fixture()
        raw.loc[raw['Tasks'].eq(60) & raw['Study Stage'].eq('Warm'), 'Optimal Certified'] = False
        data = prepare_paper_data(raw, manifest, experiment='B')
        last = data['stats'][data['stats']['Tasks'].eq(60)]
        self.assertEqual(len(data['references']), 120)
        self.assertTrue(last['n_known_optima'].eq(0).all())
        self.assertTrue(last['shortfall_mean_pct'].isna().all())
        self.assertTrue(last['benefit_mean'].notna().all())
        with tempfile.TemporaryDirectory() as directory:
            section_fragment(data, Path(directory), False)
            self.assertIn('120/150', (Path(directory) / 'Section_4_2.tex').read_text())

    def test_pilot_b_has_no_final_section(self):
        raw, manifest = b_fixture('pilot')
        data = prepare_paper_data(raw, manifest, True, experiment='B')
        with tempfile.TemporaryDirectory() as directory:
            section_fragment(data, Path(directory), True)
            self.assertFalse((Path(directory) / 'Section_4_2.tex').exists())

    def test_attribution_preserves_losses_ties_and_paired_denominators(self):
        raw, manifest = fixture('full')
        data = prepare_paper_data(raw, manifest)
        counts = attribution_counts(data)
        self.assertTrue((counts['wins'] + counts['ties'] + counts['losses']).eq(counts['n']).all())
        self.assertTrue(counts['median_gain'].eq(0).all())
        gains = data['gains'].copy()
        gains.loc[gains.index[0], 'Injection gain'] = -.03
        counts = attribution_counts({**data, 'gains': gains})
        self.assertEqual(counts['losses'].sum(), 1)
        self.assertLess(counts['mean_gain'].min(), 0)

    def test_vector_exports_and_raster_memory_guard(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory, plt.rc_context(PUBLICATION_STYLE):
            p = Path(directory)
            fig, ax = plt.subplots(figsize=(2., 1.))
            ax.plot([1, 5, 15], [0, .2, .8])
            _save(fig, p, 'quality', 100)
            for extension in ('pdf', 'svg', 'png'):
                self.assertTrue((p / ('quality.' + extension)).is_file())
            self.assertIn('<svg', (p / 'quality.svg').read_text())
            fig, ax = plt.subplots(figsize=(20, 20))
            with self.assertRaisesRegex(ValueError, '512 MiB'):
                _save(fig, p, 'too_large', 2000)
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
