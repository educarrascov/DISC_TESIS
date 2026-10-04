"""Regression checks for missing values, matched gaps and warm-start attribution.

Run from the companion directory. TTAP_TEST_A/B can override historical paths.
The committed one-instance pilot is used only as a regression fixture.
"""

import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from figure5_and_analysis import prepare, attach_references, summarize
from visual_options import ROOT, A_RELATIVE, B_RELATIVE, load_data
from warm_visual_options import prepare_warm


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a = Path(os.environ.get("TTAP_TEST_A", ROOT.parent / A_RELATIVE))
        b = Path(os.environ.get("TTAP_TEST_B", ROOT.parent / B_RELATIVE))
        cls.historical, cls.reference = load_data(a, b)
        pilot = ROOT / "examples" / "functional_pilot_n1_20261004"
        cls.raw = pd.read_csv(pilot / "metrics.csv")
        cls.trace = pd.read_csv(pilot / "trace.csv")

    def test_original_cohorts_and_reference_coverage(self):
        counts = self.reference.groupby("Configuration").size().to_dict()
        self.assertEqual(counts, {"A": 150, "B": 117})
        detail = summarize(self.historical)
        self.assertTrue(detail["instances"].eq(30).all())
        missing = self.historical[~self.historical["Has Incumbent"]]
        self.assertTrue(missing["Incumbent value"].isna().all())
        self.assertTrue(missing["Incumbent shortfall pct"].isna().all())
        self.assertTrue(missing["Available value"].eq(0).all())

    def test_paired_shortfall_is_not_a_ratio_of_sample_means(self):
        raw = pd.DataFrame({"Configuration": ["A"] * 3,
                            "Scenario": ["large", "small", "missing"],
                            "Tasks": [20] * 3, "Seed": [1, 2, 3],
                            "Method": ["HiGHS-1s"] * 3, "Time Limit sec": [1.] * 3,
                            "Has Incumbent": [True, True, False],
                            "Optimal Certified": [False] * 3,
                            "Normalized Mission Value": [.4, .2, np.nan]})
        refs = raw[["Configuration", "Scenario", "Tasks"]].copy()
        refs["Certified optimum"] = [.8, .2, .6]
        refs["Reference budget sec"] = 600
        linked = attach_references(prepare(raw), refs)
        result = summarize(linked, include_tasks=False).iloc[0]
        self.assertAlmostEqual(result["mean_incumbent"], .3)
        self.assertAlmostEqual(result["mean_incumbent_shortfall_pct"], 25.)
        self.assertAlmostEqual(result["mean_available_shortfall_pct"], 50.)
        self.assertEqual(result["matched_incumbents"], 2)
        self.assertEqual(result["matched_certified_references"], 3)

    def test_pilot_separates_optimization_from_injection(self):
        new, initial, gain = prepare_warm(self.raw, self.reference, allow_partial=True)
        self.assertEqual(len(new), 24)
        self.assertEqual(len(initial), 4)
        self.assertTrue(np.allclose(gain["Injection gain over external fallback"], 0, atol=1e-8))
        a = gain[gain["Configuration"].eq("A") & gain["Budget sec"].eq(15)]
        self.assertTrue(a["Gain over received greedy"].gt(.05).all())
        b = gain[gain["Configuration"].eq("B")]
        self.assertTrue(np.allclose(b["Gain over received greedy"], 0, atol=1e-8))
        with self.assertRaisesRegex(ValueError, "incomplete paper cohort"):
            prepare_warm(self.raw, self.reference)

    def test_rejects_unmatched_models_and_ambiguous_starts(self):
        for column, replacement, message in [
                ("Model SHA256", "different-model", "Unmatched Model"),
                ("Objective Version", "different-objective", "Changed objective"),
                ("Source Commit", "different-source", "Changed model source"),
                ("Warm Start Value", .1, "Submitted start differs")]:
            data = self.raw.copy()
            index = data.index[data["Method"].eq("HiGHS-1s-GBStart")][0]
            data.loc[index, column] = replacement
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, message):
                prepare_warm(data, self.reference, allow_partial=True)

    def test_a_worse_injected_plan_retains_a_negative_comparison(self):
        data = self.raw.copy()
        index = data.index[data["Configuration"].eq("A") &
                           data["Method"].eq("HiGHS-1s-Cold")][0]
        data.loc[index, ["Has Incumbent", "Solver Has Incumbent"]] = True
        data.loc[index, ["Normalized Mission Value", "Solver Incumbent Value"]] = .50
        _, _, gain = prepare_warm(data, self.reference, allow_partial=True)
        rows = gain[gain["Configuration"].eq("A") & gain["Budget sec"].eq(1)]
        self.assertTrue(rows["Injection gain over external fallback"].lt(0).all())

    def test_callback_history_credits_the_initial_plan_separately(self):
        new, _, _ = prepare_warm(self.raw, self.reference, allow_partial=True)
        lookup = new.set_index(["Configuration", "Scenario", "Start Method", "Budget sec"])
        for keys, group in self.trace.groupby(
                ["Configuration", "Scenario", "Start Method", "Budget sec"], sort=False):
            times = group["Elapsed solve wall sec"].to_numpy()
            values = group["Available Value"].to_numpy()
            self.assertEqual(times[0], 0.)
            self.assertTrue(np.all(np.diff(times) >= 0))
            self.assertTrue(np.all(np.diff(values) >= -1e-8))
            self.assertTrue(pd.isna(group.iloc[0]["Solver Incumbent Value"]))
            self.assertAlmostEqual(values[-1], lookup.loc[keys, "Available value"], places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
