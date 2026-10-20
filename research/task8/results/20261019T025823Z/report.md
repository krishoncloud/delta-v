# Delta-V Task 8 measured closed-loop report

Protocol: df369db23d69771ad092d5413b07afac3a670f2d217ebc9697d6880bfc256424

**TOTAL-LABEL SAVINGS VERSUS GP-ACTIVE NOT ESTABLISHED UNDER THE LOCKED RULE**

## Complete-workflow outcome

| policy | valid_stops | premature_stops | never_stopped | median_labels_at_proposed_stop |
| --- | --- | --- | --- | --- |
| fno_active | 3 | 0 | 0 | 75.0 |
| fno_lhs | 3 | 0 | 0 | 75.0 |
| gp_active | 3 | 0 | 0 | 75.0 |
| gp_random | 3 | 0 | 0 | 75.0 |
| gp_sobol | 3 | 0 | 0 | 75.0 |

Primary predeclared comparator: GP-active. Median labels saved among comparable repetitions: 0.0; all paired repetitions comparable: True.
FNO-active versus FNO-LHS selection-attribution screen: False.
All baseline comparisons remain in paired_savings.csv and comparison_verdicts.json. A three-repetition descriptive screen is not a population guarantee or proof that this beats every GP or acquisition method.

## What ran and what stopping means

3 paired repetitions, five workflows, 4 budgets from eight through 32, batches of 8, 3 FNO members per neural workflow, 3000 optimizer steps per member per budget. Members are refit from the same source, not treated as independent customer studies. GP receives scalar labels; FNO additionally receives completed fields.
The stopping rule used calibrated interval widths on label-free probe cases. Test labels were revealed only after complete trajectories and first-stop decisions were sealed. A premature first stop remains a failure; an oracle-selected later budget cannot replace it. Post-stop shadow continuation exists to measure curves/regressions, and its actual benchmark cost remains recorded.

## Total-label and compute accounting

Each workflow starts with 8 training + 12 development + 39 calibration = 59 labeled settings. At ceiling it consumes 83. The 200 test cases per repetition are separate benchmark evaluation cost. Shared caches do not discount customer labels. Probe cases have no solved labels.
customer_prefix_costs.csv estimates standalone prefixes, including solver overhead and recorded loop/failed-attempt costs. physical_solver_executions.csv counts actual cached unique solver calls. all_attempt_costs.csv includes successful/failed fits. Source load metadata and numerical audits are separate; source-pretraining cost is unknown.
Breakeven customer-simulation seconds are a scenario calculation in breakeven.csv. This generated solver is cheap: label savings do not establish compute-time or monetary savings. Model loading, hashing and host storage speed vary. Runtime after a hard process kill may be unknown; benchmark wall time is not reconstructed by summing overlapping/shared-cache counterfactuals.

## Uncertainty and scope limits

Round-specific joint90 coverage, width, point error and subgroup estimates are in round_metrics.csv, scored_test_predictions.csv and strata.csv. Reusing independent calibration over a fixed calibration-independent acquisition path does not establish simultaneous, conditional-region, adaptive-query or stopping-time coverage guarantees. The stopping rule is a measured heuristic. It must not be marketed as a guaranteed-safe stopping policy. High forcing and low initial mix were prespecified weak-region checks following Task7.
There is one 64×64 fixed periodic fluid family, one goal, one active heuristic and a limited number of paired repetitions. Fresh controls are not a new physics family. Candidate nearest-matching changes exact Sobol/LHS design properties. Field supervision, ensemble compute cost and missing POD/field-feature baselines limit architecture attribution. Three-member uncertainty is noisier than the earlier five-member ensemble; eight-run batches provide coarser stopping resolution. The present calibration/evaluation measures those tradeoffs rather than assuming Task 7 coverage transfers. No arbitrary geometry/3D/customer/OpenFOAM capability is established.

## Reproduction, storage and next work

All designs, acquired IDs, query predictions, calibration rules, stopping records, training traces, costs, environment, code/protocol hashes and reports are retained. First proposed-stop and latest FNO weights are preserved; superseded intermediate weights are pruned with hashes/recipes after a later complete round. Intermediate models require retraining to reconstruct. Retained weights download separately from the evidence ZIP.
Task8 evaluation is complete when all planned paths, integrity checks, independent evaluation and this report exist; artifact delivery completes after packaging verification. A positive run-savings verdict is separate from completion. Task9 should test the real solver/customer workflow. Task6c-B source-diversity scaling remains deferred, and Task15 privacy separation remains required before launch. No production API or website was deployed here.
