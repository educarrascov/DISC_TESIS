# TTAP: greedy MIP starts and budget-quality reporting

This companion is based on commit `07dd90042822e2cab84209fe81c9f31ecfe230a8` of
`educarrascov/DISC_TESIS`. It preserves the implemented single-primary-output
objective, references `(3, 3, 500)`, generators, greedy rules, time grid, and MILP.
It does not alter the original notebooks or completed CSV files.

## What is and is not a warm start

The completed A optimum is a **quality reference**, not a fair initial solution
for a short-budget run on that same instance. The primary experiment submits
the feasible routes produced by Greedy Benefit or Greedy Density to HiGHS.
HiGHS may change every decision. No variable is fixed to the greedy value.

Warm starts and time-limited/anytime evaluation are compatible: every start
treatment is tested at the same short solver budgets. The budgets here are
independent restarts, not checkpoints from one persistent search.

The three requested existing-data indicators require no new solver runs:

* proportion of runs with an incumbent, and proportion with positive value;
* mean/median incumbent, with the number of available incumbents;
* paired difference to the final certified reference, where one exists.

Only the warm-start comparison needs new paired runs if the objective is kept.
Changing to a multi-output objective or changing relative weights/references
requires new optimization runs and new certified references.

## Provenance and integration points

The `20260828_..._recovered.ipynb` files contain recovery/plotting code, not
the greedy or exact solver. Do not paste the warm-start solver into their
recovery cells or overwrite their source CSVs.

* `frozen_generator_A.py`: definitions/constants from cells 1 and 2 of
  `20260828_Experimento_A_4x4_Impacto_Anytime_Trazabilidad_recovered.ipynb`,
  blob `3e0ff28b7cb6fac776a3c74387f2858ca96808af`.
* `frozen_generator_B.py`: corresponding B definitions,
  blob `d53cd2897257657e8c36f19941b2b887d1ad1392`.
* `frozen_ttap_core.py`: function definitions from sections 04, 06, 08 and 09
  of `20260824_Experimento_B_6x7_Impacto_Anytime_Trazabilidad.ipynb`,
  blob `9ca9d99e21fcc6c23ab121039f227aaa4c6a9ea3`.
  Top-level notebook tests, plotting calls and experiment runners were omitted.
  The old SciPy solver wrapper was omitted because it has no public MIP-start
  argument. The shared mathematical functions also support A's generator.
* `ttap_warm_start.py`: native HiGHS interface, start mapping, validation,
  paired experiment runner and checkpoint exports.
* `figure5_and_analysis.py`: budget summaries and figures from existing/new CSVs.

The generated network is canonically ordered before matrix construction.
This does not change feasible solutions or objective coefficients, but it can
change search order relative to historical runs. Therefore **rerun the cold
control with the same new interface and settings as the warm treatments**.
Do not attribute speed differences between historical SciPy and new highspy
runs solely to the initial solution.

For an in-notebook implementation instead, add the native solver after
`prepare_exact_problem` in section 08 of the original executable notebooks,
and replace the budget loop in `run_single_scenario` (section 09).
The state-arc matrix and objective need no warm-start reformulation.
Keeping this companion separate is less disruptive.

## Installation and correctness checks

Use a separate Python 3.12 environment. The tested package versions are pinned.

```bash
python -m venv .venv-ttap-warm
source .venv-ttap-warm/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
```

Tests compare the native model against SciPy on small instances, validate
idle and greedy start vectors, reject invalid starts, and reproduce the
historical greedy values for A/B with 20 tasks and seed 1.

Two additional functional runs used the full A/B 20-task seed-1 instances at
1 and 5 s. Their starts were feasible and were received by HiGHS. These checks
are **not** the 30-replicate paper experiment and do not establish general
performance benefits.

An additional local regression reproduced all 600 historical greedy objective
rows (two policies, 150 instances, two configurations), with maximum absolute
differences below `4e-16`. This checks generator/objective fidelity, not new
solver performance.

`20261003_TTAP_WarmStart_A_B.ipynb` is a thin notebook front end to the same
scripts. It defaults to `RUN_EXPERIMENT=False` and a small pilot. It can rebuild
the existing Figure 5 without launching a full solver experiment.

## Pilot, then full paired experiment

From this directory, begin with a small pilot:

```bash
python ttap_warm_start.py --experiments A B --tasks 20 --seeds 1 2 3 --budgets 1 5 15 60 --output runs/pilot
```

For the original 150 instances per configuration, in Bash:

```bash
python ttap_warm_start.py --experiments A --tasks 20 30 40 50 60 --seeds {1..30} --budgets 1 5 15 60 --output runs/A_full
python ttap_warm_start.py --experiments B --tasks 20 30 40 50 60 --seeds {1..30} --budgets 1 5 15 60 --output runs/B_full
```

The default is one thread, solver random seed 0, and MIP relative tolerance
`1e-6`. Each configuration involves 1,800 solver runs. The sum of nominal
solver allowances is 10.125 hours per configuration (20.25 hours for A and B), before preparation overhead
and any solver time-limit overrun; early termination can reduce actual time.
B can require substantial RAM. Do not launch full configurations in parallel
on a memory-constrained machine. Record hardware, package versions and actual
times. The budget is the allowance applied to `highs.run`, not a guaranteed
end-to-end operational deadline.

Use `--resume` with **exactly the same command and output directory** to retain
fully completed instances. Changed settings or source/solver versions are
rejected. An interrupted partial instance is rerun. Existing nonempty output
directories are never silently reused without `--resume`.

Every completed instance writes:

* `metrics.csv` and a partial checkpoint after every solver run;
* `routes.csv`, with full task/return logs for replay and route/Gantt figures;
* selected state-arc indices in `.npz`, with model and scenario fingerprints.

The root aggregates the instance metrics and stores the execution manifest.
Row bounds, variable bounds, integrality, route recovery and objective/log
agreement are checked before a solution is reported. Sparse column indices
must never be reused with a different matrix ordering or model fingerprint.

## Rebuild Figure 5 now

Use these exact historical files from the repository, not a partial/quick CSV:

* A: `../ttap_experiment_A_4x4_impact_anytime_traced/metrics_A4x4_paper_20260808_112401-0400_matched_v3_impact_anytime.csv`
* B: `../ttap_experiment_B_6x7_stress_impact_anytime_traced/metrics_B6x7_stress_paper_20260825_195505-0400_matched_v3_impact_anytime_old6x7_operational_config.csv`

The new `visual_options.py` offers five alternatives to the earlier heatmap.
Each uses the same CSVs and exports PNG/PDF figures plus the underlying paired
and summary data. No solver run is needed:

```bash
python visual_options.py --option all --output figures/options
```

| Option | Layout | Main use |
| --- | --- | --- |
| `1` | Original five certification panels, plus three separate companion figures | Preserve the original Figure 5; choose a companion or table for the paper |
| `2` | Four large A/B line panels, pooling task counts | Clear overview for an email; task-size effects are pooled |
| `3` | One card per task count, with separate availability/value/shortfall axes | Inspect how each size behaves; taller than the original figure |
| `4` | Availability/certification bars and incumbent/shortfall distributions | Inspect variability and zero-value incumbents; useful for an appendix |
| `5` | Original five certification panels above a short numerical table | Recommended compact alternative for the paper |

To generate only the selected option, use `--option 5`, for example. Use
`--stat median` to replace the conditional means with medians in options 1,
2, 3 and 5. Option 4 always displays both, with its distribution.
Option 5 also exports a booktabs `option5_table.tex`; add `\usepackage{booktabs}`
if it is not already loaded. Its certification figure can be used separately
from the table by selecting option 1's main image.

The combined-analysis notebook's AB1 plotting cell can call this script after
its data-loading cell. Keep the historical CSVs unchanged. The new notebook
`20261004_Figure5_Visual_Options.ipynb` provides a selector and separate controls
for historical charts, a functional warm-start example, and a new experiment.

These are independent restarts. A line joining budget summaries does not
reconstruct one search. Missing incumbents remain missing in the mean/median
incumbent and its shortfall. At 1 s, A's zero mean shortfall uses only two
returned incumbents out of 150 runs. The figure's availability row and the
CSV sample counts are essential to interpret this result.

### Warm-start plots and attribution

Run the matched cold, GB-start and GD-start experiment on the same machine,
then use `warm_visual_options.py`. Historical optima are quality references;
historical cold runtimes are not controls for a new warm run.

```bash
python warm_visual_options.py --warm-csv runs/A_full/metrics.csv runs/B_full/metrics.csv --output figures/warm
```

Three comparisons are generated:

1. `warm1_greedy_and_higher_budget_plans`: solver-returned incumbent rate,
   available benefit and known-optimum shortfall. The two pure greedy values
   appear as dashed baselines. A validated supplied start guarantees plan
   availability even if search adds no benefit.
2. `warm2_credit_to_greedy_vs_injection`: warm value minus its supplied greedy;
   and warm value minus `max(new cold value, same greedy)`. The second control
   asks whether injection helps beyond retaining that greedy externally while
   running HiGHS without a start. Negative differences are retained.
3. `warm3_setup_cost_and_quality`: measured preparation/start/solve costs and
   quality against actual measured time. Solver allowances are not guaranteed
   response deadlines. The timing excludes later decoding/reporting and
   fingerprint overhead.

The default analysis requires all five task counts and 30 seeds at 1, 5, 15
and 60 s for each configuration supplied. Partial cohorts require
`--allow-partial` and are labelled as functional pilots. The source commit,
objective, model/scenario fingerprints and solver settings are checked.
Sample counts and paired differences are exported for subsequent inference.

For a reproducible view of the completed one-instance pilot:

```bash
python warm_visual_options.py --warm-csv examples/functional_pilot_n1_20261004/metrics.csv --allow-partial --trace-csv examples/functional_pilot_n1_20261004/trace.csv --trace-scenario A4x4_T20_S1 --trace-budget 60 --output figures/pilot
python test_visual_options.py
```

That pilot has **one seed and 20 tasks per configuration**, with 24 solver runs.
A reaches the same certified optimum in all three treatments at 15 and 60 s.
B's starts retain their greedy values and the cold runs return no incumbent.
The incremental benefit of injection against cold plus the same greedy is
zero at every tested allowance. These checks validate the method; they do not
establish a general warm-start advantage. B's 60 s allowance also overran in
this execution: inspect the measured times rather than treating it as a hard
deadline.

### True within-run incumbent history

Use `--trace` when launching a new experiment to log the native improving-MIP
callback. It adds a `trace.csv` per instance and an aggregate at the output
root. An explicit initial event credits the supplied greedy or idle fallback;
it does not credit HiGHS with discovering that plan at time zero.

```bash
python ttap_warm_start.py --experiments A B --tasks 20 --seeds 1 --budgets 1 5 15 60 --trace --output runs/new_trace_pilot
python warm_visual_options.py --warm-csv runs/new_trace_pilot/metrics.csv --allow-partial --trace-csv runs/new_trace_pilot/trace.csv --trace-scenario A4x4_T20_S1 --trace-budget 60 --output figures/new_trace_pilot
```

`warm4_true_incumbent_trace` shows one instance and one run per treatment.
Its clock starts after model/start preparation. The trace is a separate view
from independently restarted budget summaries. See the official
[HiGHS callback documentation](https://ergo-code.github.io/HiGHS/dev/callbacks/).

`--trace` changes the execution manifest. Use a new output directory rather
than attempting to resume an old run with a different setting.

The older `figure5_and_analysis.py` remains available for reproducing the
previous heatmaps and is shared by the new statistical scripts:

```bash
python figure5_and_analysis.py --a ../ttap_experiment_A_4x4_impact_anytime_traced/metrics_A4x4_paper_20260808_112401-0400_matched_v3_impact_anytime.csv --b ../ttap_experiment_B_6x7_stress_impact_anytime_traced/metrics_B6x7_stress_paper_20260825_195505-0400_matched_v3_impact_anytime_old6x7_operational_config.csv --output figures/current
```

The older script also accepts `--warm-csv` and exports its original warm
overview and per-task summaries. Use `warm_visual_options.py` for the current
comparison: it additionally separates improvement over the greedy from the
incremental benefit of injection against a cold run with that same fallback.

For the combined-analysis notebook
`20260828_Comparacion_A_B_4x4_vs_6x7_Presupuesto_Complejidad.ipynb`, replace
the certification-only AB1/Figure-5 plotting cell with the new figure or call
this script after its input-loading cell. Keep the original certification
summary for reproducibility. Use LaTeX labels rather than hard-coded figure
numbers if a new warm-start figure changes the numbering of the GPS figures.

## Reporting definitions

`Has Incumbent` in the new runner means a validated plan is available, either
from HiGHS or the supplied feasible start. `Solver Has Incumbent` separately
records whether HiGHS actually returned one. `Available Plan Source` identifies
the difference. An accepted API submission does not by itself prove solver
incumbent availability; the submitted vector is independently validated.

The historical CSV has no external-start fallback, so its `Has Incumbent`
continues to mean a solver incumbent. A known idle fallback has value zero,
but must not be counted as a solver-returned solution.

Summary CSVs retain both:

* mean/median of **recorded available incumbents**, excluding missing values;
* mean/median **available mission value**, assigning zero when none was recorded.

These are different statistics. Conditional means can change because additional
hard instances enter the sample, not because an individual solution deteriorates.

The certified references are A at 600 s (150 instances) and B at 1,800 s
(117 instances; by task count: 30, 30, 27, 21, 9). Optimality is relative to
the declared solver tolerances. The paired shortfall is computed per instance
before aggregation. Missing incumbents have no incumbent-specific gap;
their **available-value** shortfall is 100% against a positive certified optimum.
Both statistics and their sample sizes are exported. B's remaining instances
must not be described as having a known gap to the true optimum. They can
still contribute to plan availability, value and solver-bound reporting.

Timing columns separate greedy generation, mapping, native-model conversion,
model loading, start submission and solve time. `End-to-end sec` sums these
with graph/matrix construction for a single deployment. It excludes later
decoding/reporting and fingerprint overhead. Start setup is not treated as
free when discussing end-to-end response times.

## Optional replay of a previously computed incumbent

Metrics alone are not a MIP start: a scalar objective does not specify a route
or decision vector. The repository also contains task/return logs. The runner
can replay them **on the same instance** using `--replay-logs`,
`--replay-metrics` and `--replay-method`. It verifies the route and objective.

Example for an earlier A incumbent:

```bash
python ttap_warm_start.py --experiments A --tasks 20 --seeds 1 --budgets 1 5 --output runs/A_replay --replay-logs ../ttap_experiment_A_4x4_impact_anytime_traced/task_logs_A4x4_paper_20260808_112401-0400_matched_v3_impact_anytime.csv --replay-metrics ../ttap_experiment_A_4x4_impact_anytime_traced/metrics_A4x4_paper_20260808_112401-0400_matched_v3_impact_anytime.csv --replay-method HiGHS-15s --allow-oracle-replay
```

For this particular A/20/seed-1 instance, 15 s already certified the optimum,
so it is explicitly an **oracle/implementation check**, not evidence that
a practical quick heuristic can match an optimum. To study staged optimization,
choose an earlier feasible **uncertified** incumbent, omit the oracle flag,
charge its prior solve time and compare against a continuous search with the
same total budget. `Sequential Solver Budget sec` and total sequential time
are retained. The current experiment performs warm restarts, not continuation
of the previous branch-and-bound tree.

If the selected earlier run has no incumbent, replay is rejected. An A route
cannot be injected into B unchanged: nodes, travel times and fleet capacities
differ. That would require a separate transfer-and-repair experiment, which is
not needed to address Pedro's warm-start request.

## Effect on existing paper figures

Do not add warm treatments indiscriminately to every old plot.

* Keep A/B quality and runtime tables and their full-budget figures as the
  original benchmark, if the objective and instances are unchanged.
* Preserve Figure 5's certification curves; choose a companion figure or the
  compact table from options 1/5 for availability, value and known-optimum shortfall.
* Add one compact warm-start comparison subsection/figure after that analysis.
* Network-size figures do not change: initial solutions do not add variables
  or constraints.
* Route/Gantt/GPS figures need new treatments only if those cases are rerun.

References used to select the software interface:
[SciPy milp API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html),
[HiGHS Python API](https://ergo-code.github.io/HiGHS/dev/interfaces/python/example-py/).
