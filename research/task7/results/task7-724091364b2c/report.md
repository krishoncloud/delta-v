# Delta-V Task 7 measured uncertainty report

Protocol: 724091364b2c1a44cd41f2aaa554e79e3fca297e2eb92c24477afd26dbf4d4a2

**MEETS THE PRESPECIFIED EMPIRICAL RESEARCH SCREEN**

Primary: pretrained five-or-more-member FNO, 16 supplied training settings, normalized joint split-conformal intervals at nominal 90%.
Observed joint coverage: 0.915; Wilson 95% interval [0.868, 0.946] across 200 independent test settings.
Mean widths: {'kinetic_energy': 0.01461733059541376, 'pressure_rms': 0.012681703556608234}. Provisional width ceiling: [0.03, 0.03].
The coverage screen allows five percentage points below nominal; passing is an empirical research screen, not proof of conditional calibration or engineering safety.

## What executed

4 nested supplied-data budgets, 5 full-tuned members per budget, one fixed supplied-data ordering, and a development-selected GP at each budget. All methods/budgets/levels were frozen before calibration/test answers.
Coverage, full width, interval score, point RMSE, outcome-specific/joint results, 11 predefined parameter/edge strata, and uncertainty-bin/error-correlation diagnostics are retained in CSV files. Raw FNO/GP, calibrated marginal/joint FNO/GP, and constant-width FNO comparisons all remain visible.

## Data and computation cost

At the primary budget this workflow consumed 16 training + 12 development + 39 calibration = **67 labeled settings**, plus 200 final-test numerical references used for this benchmark. The union across all budget experiments used 32 training settings, not the sum of nested budgets. Every ensemble member reuses the same labels at its budget.
All recorded successful/failed fit attempts total 1061.5 seconds. This excludes source-pretraining cost, whose complete cost is unknown. Source load, audit, solver reference costs, file sizes and environment are preserved separately; benchmark test simulations are not hidden as free product inputs.

## What the result means

Calibrated intervals have marginal-over-settings split-conformal validity under exchangeability and a predictor/score frozen independently of calibration. Joint intervals target both declared scalar outcomes together. There is no per-setting, conditional-region, simultaneous-all-models, fieldwise or shifted-distribution guarantee. Held-out finite-sample coverage varies; Wilson intervals display that uncertainty. Stratified analyses are descriptive and correlated, not many independent successes.
This is one generated periodic fluid family, one training subset ordering and a limited ensemble. All members share the same pretrained backbone, so disagreement can miss shared bias. The source systems are known, but a complete legacy exposure ledger is unavailable. It is not a customer/OpenFOAM validation or a demonstration of prospective solver savings. More pretraining/source diversity remains deferred Task 6c-B work.

## Completion and next step

Task 7 scientific deliverable is complete when the full campaign, prediction/rule seals, calibration, final testing, report and verified archive exist. Interval usability is reported separately from completion. Task 8 must count development/calibration overhead, test acquisition under changing data, and address the fact that adaptively chosen queries need not follow this experiment's calibration distribution. These intervals do not establish a safe stopping policy. Task 15 private storage remains required before launch.
