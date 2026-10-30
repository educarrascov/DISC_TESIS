Source: one actual 600-s search per instance and initialization.
Only callbacks recorded at or before each elapsed-time point contribute.
Validated greedy plans are credited separately at solve time zero.
Cold without an observed incumbent is scored separately as idle=0 for available benefit;
its observed incumbent remains missing. Missing observed incumbents are not backfilled.
Preparation time is excluded. Callbacks document observations, not necessarily the earliest
internal discovery time. Certification is counted only after an optimal run finished.
Post-hoc gaps use certified references of the same model/objective only.
This history is distinct from independent restarts at nominal HiGHS allowances.
