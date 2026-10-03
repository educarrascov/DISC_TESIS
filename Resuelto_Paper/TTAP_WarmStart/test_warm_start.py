"""Correctness checks; not the 30-replicate paper experiment."""

import math
import unittest

import numpy as np
from scipy.optimize import milp

from ttap_warm_start import load_kernel, logs_to_start_vectors, solve_one, to_highs_lp, validate_vector


class WarmStartTests(unittest.TestCase):
    def tiny_scenario(self, configuration):
        kernel = load_kernel(configuration)
        scenario = kernel["create_scenario"](20, 1)
        scenario["tasks"] = {key: value for key, value in scenario["tasks"].items() if key <= 2}
        scenario["task_to_node"] = {key: value for key, value in scenario["task_to_node"].items() if key <= 2}
        scenario["params"]["TMAX"] = 80.0
        kernel["assign_task_value_metadata"](scenario)
        return kernel, scenario

    def test_native_model_equivalence_and_starts(self):
        for configuration in ["A", "B"]:
            with self.subTest(configuration=configuration):
                kernel, scenario = self.tiny_scenario(configuration)
                package = kernel["prepare_exact_problem"](scenario)
                logs = {label: kernel["solve_greedy_ttap"](scenario, rule=rule)[0]
                        for label, rule in [("GBStart", "benefit"), ("GDStart", "density_operational")]}
                starts = logs_to_start_vectors(kernel, scenario, package, logs)
                reference = milp(c=package["c"], integrality=package["integrality"],
                                 bounds=package["bounds"], constraints=package["constraints"],
                                 options={"time_limit": 10, "mip_rel_gap": 1e-6})
                self.assertTrue(reference.success)
                lp = to_highs_lp(package)
                for label in ["Cold", "GBStart", "GDStart"]:
                    result, summary = solve_one(scenario, package, lp, 10, label,
                                                start=starts.get(label))
                    self.assertTrue(summary["Optimal Certified"])
                    self.assertTrue(math.isclose(result.fun, reference.fun, abs_tol=1e-8))
                    kernel["validate_exact_incumbent"](result, summary, package)
                    if label != "Cold":
                        self.assertTrue(summary["Warm Start Submitted"])
                        self.assertGreaterEqual(-result.fun + 1e-9, summary["Warm Start Value"])

    def test_idle_start_and_rejected_invalid_start(self):
        kernel, scenario = self.tiny_scenario("A")
        package = kernel["prepare_exact_problem"](scenario)
        start = logs_to_start_vectors(kernel, scenario, package, {"Idle": None})["Idle"]
        self.assertEqual(-float(package["c"] @ start), 0.0)
        validate_vector(package, start)
        with self.assertRaises(ValueError):
            validate_vector(package, np.zeros_like(start))
        with self.assertRaises(ValueError):
            validate_vector(package, start + .25)

    def test_original_greedy_values(self):
        expected = {"A": {"benefit": .456290, "density_operational": .470814},
                    "B": {"benefit": .621891, "density_operational": .683246}}
        for configuration, rules in expected.items():
            kernel = load_kernel(configuration)
            scenario = kernel["create_scenario"](20, 1)
            for rule, value in rules.items():
                frame, _ = kernel["solve_greedy_ttap"](scenario, rule=rule)
                self.assertAlmostEqual(float(frame["Benefit"].sum()), value, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
