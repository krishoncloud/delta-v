# Delta-V Task 6 — measured report

Protocol: `7fb9154efb8a8c0eae581c92d2c024d918c0548cdb1210ba64fa33ffec7b7ee2`. Study: `forced_multimode_periodic_ns_v1`.

## Primary product result: fewer supplied runs
win on this study: 16 versus 24 supplied settings. Qualification requires joint outcome success >= 0.9, maximum tolerance ratio <= 2.0, complete coverage, and attainment on every seed. These are the smallest **tested** budgets, not interpolated counts.

| method | smallest_tested_budget | status | validation_settings | confirmation_settings |
| --- | --- | --- | --- | --- |
| mean | nan | not reached | 12 | 24 |
| linear | nan | not reached | 12 | 24 |
| gp | 24.0 | reached | 12 | 24 |
| scratch | 24.0 | reached | 12 | 24 |
| pretrained_full | 16.0 | reached | 12 | 24 |
| pretrained_frozen | nan | not reached | 12 | 24 |

## Does pretraining help?
not established under locked rule; inspect paired comparisons. Primary budgets: [8, 16]. Paired hierarchical confidence intervals are descriptive with only 3 seeds. No individual pixels were counted as independent samples. Development choices used validation and are not fresh confirmatory evidence.

## FNO versus GP: where it wins, ties or loses
| opportunity | verdict |
| --- | --- |
| new audited fluid study | win on this study: 16 versus 24 supplied settings |
| spatial engineering outcomes | win on this study: 16 versus 24 supplied settings; fields additionally supplied to FNO |
| fewer supplied settings | win on this study: 16 versus 24 supplied settings; retrospective only |
| many controls (~10+) | not tested: three-control candidate |
| regime transition | not tested: no certified transition |
| OpenFOAM/customer production | not tested: Task 9 |

| baseline | budget | stratum | ratio | ci_low | ci_high | settings | seeds | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gp | 8 | all | 0.31639016793082325 | 0.22018836589918736 | 0.4131018167911994 | 24 | 3 | win |
| scratch | 8 | all | 0.9815673604938883 | 0.6103375691176258 | 1.3935750138599043 | 24 | 3 | inconclusive |
| gp | 16 | all | 0.8555232434130859 | 0.5783438896669324 | 1.5687736982857734 | 24 | 3 | inconclusive |
| scratch | 16 | all | 0.944166399373125 | 0.7694822921566529 | 1.1741460875384047 | 24 | 3 | inconclusive |
| gp | 24 | all | 1.7583178806148705 | 1.3480702344603162 | 2.204878382162978 | 24 | 3 | loss |
| scratch | 24 | all | 0.7367341280955361 | 0.5862713672897283 | 0.8926748925181537 | 24 | 3 | win |
| gp | 32 | all | 2.862726546677545 | 2.1052527377020938 | 4.056441165470402 | 24 | 3 | loss |
| scratch | 32 | all | 0.6845706637220852 | 0.5186343836965274 | 0.8978550670364321 | 24 | 3 | win |

Every predeclared edge/interior result is in paired_comparisons.csv. Outcome-specific ratios are in fno_gp_ratios.csv; undefined zero-denominator ratios remain missing. New post-hoc findings need fresh confirmation.

## Compact adaptation
Frozen-backbone accuracy is in metrics.csv and runs_to_tolerance.csv; artifact bytes and fitting times are in fit_costs.csv. Base weights are shared but must remain available and hash-verified. The adapter here is the conditioner plus the lifting and projection layers (the spectral blocks are frozen). It saves per-study storage; it does not eliminate backbone forward/backward computation.

## What was supplied and what it cost
32 unique training-pool settings, 12 validation settings, 24 confirmation settings. No pilot was run; the training configuration was fixed before fitting but chosen after the Task 6a results (protocol config_basis). Numerical audit executions are additional and stored in protocol.json. The full benchmark acquired more settings than a single deployed adaptation. reference_costs.csv, confirmation/costs.json and all_attempt_costs.csv account for reference work and failed/retried fits. Unmeasured external generation/export/pretraining costs are unknown, never zero. Reference timer excludes some Python/export overhead; overall preparation wall time is recorded separately.

## Fairness and limitations
All six arms share training IDs at each budget/seed. FNO arms also receive fields; scalar GP receives knobs and scalar outcomes. A field-feature/POD baseline was omitted from this six-arm implementation, so the GP comparison cannot isolate the benefit of field supervision. Scratch versus pretrained uses identical information and architecture. GP is an ARD RBF/Matern scalar regressor, not a claim about all possible GP models. Deterministic baseline results on identical anchor-only subsets are not independent repetitions.

The target is a controlled generated periodic fluid study, not a customer/OpenFOAM case. Source registry exclusion is audited but the legacy checkpoint lacks a complete signed exposure ledger. Three knobs do not establish high-dimensional superiority; no regime transition is claimed. Outcome dependence and numerical convergence are recorded. Tolerance basis: These are provisional research thresholds for this synthetic benchmark, not customer-approved engineering limits. I chose absolute tolerances of 0.015 in each nondimensional outcome, zero additional relative tolerance, 90% joint per-setting success, and a maximum error of 2× tolerance to make the run-to-accuracy comparison measurable. Results will be reported against these thresholds and should not be treated as safe design decisions..

This is retrospective data efficiency, not a demonstrated online stopping policy or prospective solver savings. Uncertainty calibration remains Task 7; prospective acquisition Task 8; real solver Task 9. No website deployment or public model upload occurs. Task 15 remains required before launch.

## Completion
All six required arms completed at all locked budgets/seeds and models were frozen before confirmatory reference access. Audit failures prevent this report from being produced. A measured loss is a completed comparison. All raw metrics, selected models, configurations, warnings, and failed attempts remain in this session directory.
