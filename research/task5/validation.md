# Task 5 notebook validation — 17 October 2026

The deliverable is `research/molab_task5.py`, a self-contained marimo notebook.
Machine-readable results and exact notebook/checkpoint hashes are in
[validation.json](validation.json). No generated weights or customer fields are
published by this change.

## Final notebook checks

- `marimo check research/molab_task5.py`: no notebook diagnostics.
- Eight offline tests pass. They cover the exact-reference physics residual,
  manifest/shape/unit rejection, whole-setting leakage, train-only normalization,
  forbidden query labels and unsupported queries, execution buttons, save/reload,
  artifact hashes, recovery downloads and source-checkpoint hash rejection.
- The offline complete-stage test uses an explicitly substituted small FNO for
  speed. This substitution exists only in the test; the notebook has no such mode.
- A separate **complete run of the final notebook** used the actual source FNO on
  CPU for six steps. Source hash verification, strict loading, real backbone and
  conditioner gradients, all plots, baseline comparisons, label-free queries,
  reload and both ZIP downloads completed. The snapshot inside its archive matches
  the delivered notebook's SHA-256.
- The six-step run reduced development loss from 1.93585 to 1.42801, but did not beat
  the mean scalar predictor. It correctly reported `NEEDS WORK`. Passing an
  execution test is not the same as passing learning-quality gates.

## Additional 200-step development learning check

A direct call to the same training implementation, using the actual source FNO,
seed 42 and the 16/9 training/development split, ran for 200 steps on CPU. It took
about 258 seconds, selecting step 180 by development loss. It ran before final
validation hardening, zero-pressure RMS derivative handling and the recovery
button were added; its exact source snapshot hash is recorded separately.
The final version then received the complete source-checkpoint smoke test above.

- Initial development loss: **1.93585**.
- Selected development loss: **0.004124**.
- Mean training-field loss: **1.75121 → 0.003826**.
- Development kinetic-energy RMSE: **0.009841**.
- Development pressure-RMS error: **0.004052**.
- On the same split, mean RMSE was approximately **0.161115**, linear **0.028451**,
  and GP **0.000277** for each scalar reference.

This establishes meaningful learning of the controlled task. FNO beats mean and
linear in that check; **GP remains better**. The two reference outcomes happen to
be equal for this vortex, so these are not independent successes. Development
selected the checkpoint; these are not untouched test results. No scratch FNO
comparison was run here, so this does not isolate the value of pretraining.

## Limits

Validation ran locally on CPU, not inside the user's MoLab or on CUDA. The
notebook's GPU dependencies and execution controls are configured for MoLab, but
no claim is made that a particular remote GPU runtime was tested. The default
GPU budget is 200 steps/900 seconds. CPU UI defaults to six steps and is capped at
ten; the longer check called the offline training function directly.

The controlled analytical case is inexpensive and simple. It verifies the
parameter/initial-state → real FNO → field → engineering-outcome mechanism,
not customer simulation savings, robust new-problem transfer, calibrated
uncertainty or production readiness. Task 6 needs a harder audited study and
matched scratch/adapter/full-tuning comparisons. Task 15 remains paused and
mandatory before launch.
