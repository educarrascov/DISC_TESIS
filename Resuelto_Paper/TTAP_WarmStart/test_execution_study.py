"""Checks of value semantics, MIP starts, reference identity and checkpoints."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import milp

import ttap_warm_start as native
from objective_profiles import ObjectiveSpec, apply_objective, compute_metrics, professor_examples
from study_runner import configuration, manifest_for, StudyConfig
from study_analysis import prepare_study


class ExecutionStudyTests(unittest.TestCase):
    def reporting_fixture(self, spec):
        rows = []
        common = {"Configuration": "A", "Scenario": "A4x4_T20_S1", "Tasks": 20, "Seed": 1,
                  "Objective SHA256": spec.fingerprint, "Model SHA256": "same-model",
                  "Scenario SHA256": "same-scenario", "HiGHS Version": "1.11.0",
                  "Threads": 1, "Random Seed": 0, "End-to-end sec": 1.,
                  "Has Incumbent": True, "Solver Has Incumbent": True,
                  "Optimal Certified": False}
        for m, value, start in (("Cold", .50, np.nan), ("GBStart", .40, .40), ("GDStart", .55, .50)):
            rows.append({**common, "Study Stage": "Warm", "Method": "HiGHS-1s-"+m,
                         "Start Method": m, "Time Limit sec": 1., "Normalized Mission Value": value,
                         "Solver Incumbent Value": value, "Warm Start Value": start})
        for m, value in (("GBStart", .40), ("GDStart", .50)):
            rows.append({**common, "Study Stage": "Heuristic", "Method": "Greedy-"+m,
                         "Start Method": m, "Normalized Mission Value": value})
        rows.append({**common, "Study Stage": "Reference", "Method": "HiGHS-600s-ColdReference",
                     "Start Method": "ColdReference", "Time Limit sec": 600.,
                     "Normalized Mission Value": .70, "Solver Incumbent Value": .70, "Optimal Certified": True})
        return pd.DataFrame(rows)

    def tiny(self, config, spec):
        kernel = native.load_kernel(config)
        scenario = kernel["create_scenario"](20, 1)
        scenario["tasks"] = {i: t for i, t in scenario["tasks"].items() if i <= 2}
        scenario["task_to_node"] = {i: n for i, n in scenario["task_to_node"].items() if i <= 2}
        scenario["params"]["TMAX"] = 80.
        return kernel, apply_objective(scenario, spec)

    def test_pedros_mixed_task_example(self):
        values = professor_examples()
        self.assertAlmostEqual(values["mixed"], 19 / 6)
        self.assertAlmostEqual(values["medical_only"], 2 / 3)
        self.assertGreater(values["mixed"], values["medical_only"])
        legacy = ObjectiveSpec("primary")
        kernel = native.load_kernel("A")
        scenario = kernel["create_scenario"](20, 1)
        scenario["tasks"] = {1: {"med": 1, "pers": 3, "load": 500},
                             2: {"med": 1, "pers": 0, "load": 0}}
        apply_objective(scenario, legacy)
        self.assertAlmostEqual(scenario["tasks"][1]["potential_value"], scenario["tasks"][2]["potential_value"])

    def test_legacy_coefficients_and_resource_vectors_unchanged(self):
        for experiment in ("A", "B"):
            kernel = native.load_kernel(experiment)
            scenario = kernel["create_scenario"](20, 1)
            before = deepcopy(scenario)
            values = [t["potential_value"] for t in scenario["tasks"].values()]
            apply_objective(scenario, ObjectiveSpec("primary"))
            self.assertTrue(np.allclose(values, [t["potential_value"] for t in scenario["tasks"].values()]))
            apply_objective(scenario, ObjectiveSpec())
            for i, task in scenario["tasks"].items():
                self.assertEqual([task[x] for x in ("med", "pers", "load")],
                                 [before["tasks"][i][x] for x in ("med", "pers", "load")])
            self.assertEqual(scenario["nodes"], before["nodes"])
            self.assertEqual(scenario["helicopters"], before["helicopters"])

    def test_native_scipy_and_greedy_starts_for_both_objectives(self):
        for experiment in ("A", "B"):
            for mode in ("primary", "multidimensional"):
                kernel, scenario = self.tiny(experiment, ObjectiveSpec(mode))
                package = kernel["prepare_exact_problem"](scenario)
                reference = milp(c=package["c"], integrality=package["integrality"], bounds=package["bounds"],
                                 constraints=package["constraints"], options={"time_limit": 10., "mip_rel_gap": 1e-6})
                self.assertTrue(reference.success)
                for method, rule in (("GBStart", "benefit"), ("GDStart", "density_operational")):
                    logs, _ = kernel["solve_greedy_ttap"](scenario, rule=rule)
                    start = native.logs_to_start_vectors(kernel, scenario, package, {method: logs})[method]
                    result, summary = native.solve_one(scenario, package, native.to_highs_lp(package),
                                                       10, method, start=start, trace=[])
                    self.assertTrue(summary["Optimal Certified"])
                    self.assertAlmostEqual(result.fun, reference.fun, places=8)
                    self.assertGreaterEqual(-result.fun + 1e-8, -float(package["c"] @ start))
                with self.assertRaisesRegex(ValueError, "constraint bounds"):
                    native.solve_one(scenario, package, native.to_highs_lp(package),
                                     1, "BadStart", start=np.zeros(len(package["c"])))

    def test_multidimensional_reporting_counts_auxiliary_quantities(self):
        spec = ObjectiveSpec()
        kernel, scenario = self.tiny("A", spec)
        scenario["tasks"][1].update({"med": 1, "pers": 3, "load": 500})
        apply_objective(scenario, spec)
        frame, summary = kernel["solve_greedy_ttap"](scenario, rule="benefit")
        row = compute_metrics(kernel, scenario, frame, summary, spec)
        ids = frame.loc[frame["Status"].eq("Task Executed"), "Task ID"].astype(int)
        self.assertEqual(row["Cargo Delivered kg"], sum(scenario["tasks"][i]["load"] for i in ids))
        self.assertEqual(row["Personnel Transported"], sum(scenario["tasks"][i]["pers"] for i in ids))

    def test_manifest_json_roundtrip_and_workload(self):
        a = configuration("A", "full")
        b = configuration("B", "full")
        manifest = manifest_for(a)
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertEqual(a.workload()["warm_solver_runs"], 1800)
        self.assertAlmostEqual(a.workload()["nominal_warm_solver_hours"], 10.125)
        self.assertAlmostEqual(a.workload()["nominal_reference_solver_hours"], 25.)
        self.assertAlmostEqual(b.workload()["nominal_reference_solver_hours"], 100.)
        self.assertFalse(ObjectiveSpec().historical_compatible)
        self.assertFalse(ObjectiveSpec("primary", {"medical": 3, "personnel": 1.5, "cargo": 1}).historical_compatible)

    def test_objective_normalization_and_unchanged_feasible_network(self):
        for mode in ("primary", "multidimensional"):
            kernel, scenario = self.tiny("A", ObjectiveSpec(mode))
            value = sum(kernel["task_benefit"](scenario, i, 0) for i in scenario["tasks"])
            self.assertAlmostEqual(value, 1.)
        kernel, primary = self.tiny("A", ObjectiveSpec("primary"))
        multidimensional = deepcopy(primary)
        apply_objective(multidimensional, ObjectiveSpec())
        a, b = kernel["prepare_exact_problem"](primary), kernel["prepare_exact_problem"](multidimensional)
        self.assertEqual(a["A"].shape, b["A"].shape)
        self.assertEqual((a["A"] != b["A"]).nnz, 0)
        self.assertTrue(np.array_equal(a["constraints"].lb, b["constraints"].lb))
        self.assertTrue(np.array_equal(a["constraints"].ub, b["constraints"].ub))

    def test_talcahuano_snapshot_and_existing_benefit(self):
        from talcahuano_three_methods import gps_kernel_and_scenario
        kernel, scenario = gps_kernel_and_scenario(ObjectiveSpec("primary"))
        self.assertEqual(len(scenario["tasks"]), 30)
        self.assertEqual(len(scenario["helicopters"]), 6)
        routes, summary = kernel["solve_greedy_ttap"](scenario, rule="benefit")
        row = compute_metrics(kernel, scenario, routes, summary, ObjectiveSpec("primary"))
        self.assertAlmostEqual(row["Normalized Mission Value"], .5160779921750199, places=10)

    def test_reference_identity_and_incremental_attribution(self):
        spec = ObjectiveSpec()
        data = self.reporting_fixture(spec)
        solver, warm, _, gain, refs = prepare_study(data, spec)
        self.assertEqual(len(warm), 3)
        self.assertEqual(len(refs), 1)
        by_method = gain.set_index("Start Method")
        self.assertAlmostEqual(by_method.loc["GBStart", "Injection gain"], -.10)
        self.assertAlmostEqual(by_method.loc["GDStart", "Gain over greedy"], .05)
        old_refs = refs.copy()
        old_refs["Objective SHA256"] = ObjectiveSpec("primary").fingerprint
        with self.assertRaisesRegex(ValueError, "Historical optima are invalid"):
            prepare_study(data, spec, old_refs)
        changed = data.copy()
        changed.loc[changed["Study Stage"].eq("Reference"), "Model SHA256"] = "different-model"
        with self.assertRaisesRegex(ValueError, "Changed Model"):
            prepare_study(changed, spec)
        with self.assertRaisesRegex(ValueError, "full 30-replicate"):
            prepare_study(data, spec, require_full=True)

    def test_recorded_solver_gap_differs_from_external_fallback_gap(self):
        spec = ObjectiveSpec()
        data = self.reporting_fixture(spec)
        mask = data["Start Method"].eq("GBStart")
        data.loc[mask, "Normalized Mission Value"] = .60
        data.loc[mask & data["Study Stage"].eq("Warm"), "Warm Start Value"] = .60
        solver, _, _, _, _ = prepare_study(data, spec)
        row = solver[solver["Start Method"].eq("GBStart")].iloc[0]
        self.assertAlmostEqual(row["Available shortfall pct"], 100 / 7)
        self.assertAlmostEqual(row["Solver shortfall pct"], 300 / 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
