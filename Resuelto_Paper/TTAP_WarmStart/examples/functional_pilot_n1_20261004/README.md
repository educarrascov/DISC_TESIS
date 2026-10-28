# Functional pilot: one instance in A and one in B

These files record a real local run, not the completed paper experiment.

* 20 tasks, seed 1, both configurations.
* Independent 1, 5, 15 and 60 s solver allowances.
* Matched cold, Greedy-Benefit start and Greedy-Density start.
* Native HiGHS 1.11.0; one thread, solver seed 0; callback tracing enabled.
* The unchanged primary-output objective and frozen source commit are recorded
  in `metrics.csv` and `manifest.json`.

`metrics.csv` has four directly measured greedy rows and 24 solver rows.
`trace.csv` credits the initial external plan separately from solver events.
The manifest records the tested environment and script fingerprints.

A certified the same optimum in all three treatments at 15 and 60 s. B did
not improve either initial greedy plan within the tested runs. Injection's
incremental value against the matched cold run with the same greedy fallback
is zero in this pilot. Do not infer population-level performance from n=1.
Solver time limits are allowances: B's 60 s runs exceeded that wall time.

Reproduce its figures with the command in the parent README. Rerunning can
change search timing; retain and label each new manifest and cohort.
