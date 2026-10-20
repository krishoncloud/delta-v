# Task 7 notebook — delivery and validation

Notebook: `DeltaV_Task7_MoLab.py` (one self-contained Marimo/MoLab file, 61 cells, 19 numbered sections).

## Run on MoLab

1. Import the `.py` notebook and select a CUDA runtime. The script header declares the dependencies; the notebook records installed versions.
2. Read sections 1–3, submit the full-run configuration, then click **Prepare / resume Task 7 study and solver audit**.
3. Inspect the ranges, numerical audit and locked campaign matrix. Click **Load verified original FNO source**.
4. Click **Run / resume ALL fits**. Defaults are four budgets (8/16/24/32) with five members each, all trained for up to 3,000 optimizer steps. The four matched GP workflows tune kernels on development data. No pilot is required.
5. After all fits finish and predictions are sealed, click **Generate calibration answers and freeze interval rules**.
6. Once test intervals are sealed, click **Reveal fresh final-test answers and evaluate once**. Read the report, coverage/width tables, local diagnostics, figures and data-cost ledger.
7. Click **Verify and package complete Task 7 artifacts**. Download the evidence ZIP, all 20 member-weight ZIPs, and archive inventory before closing MoLab. The evidence plus five budget-16 weight archives suffice for the primary saved inference function; all archives preserve the full experiment.

Allow at least 15 GB free for the default run. Source weights are about 858 MB including the original training checkpoint contents; each adapted inference state is about 286 MB, so the 20 members occupy about 5.7 GB, plus a similar amount for their archives. The notebook reserves archive space in its disk check.

The same output directory and submitted configuration resume completed fits. A failed fit restarts that fit. A hard runtime kill may lose that fit's unrecorded elapsed time. Model/interval seals prevent mixing configurations. Reopening requires submitting the same directory and configuration; changing explanatory Markdown is allowed, while scientific code/configuration changes need a new run. Do not reuse inspected test cases as fresh confirmation.

Outputs are local and downloadable; this notebook does not upload proprietary weights to public Hugging Face or GitHub. The original public HF source is a read-only pinned download. Task 15's private-product-storage work remains separate.

## What the experiment covers

- Fresh generated controls in Task 6's supported fixed-geometry, periodic 64×64 fluid family.
- Identical supplied-run IDs within each ensemble and for its matched GP.
- Separate training pool (32), development (12), calibration (39), and final test (200).
- Raw FNO/GP intervals, exact split-conformal marginal/joint intervals, and a constant-width FNO reference.
- Coverage at 50/80/90/95%, Wilson confidence intervals, width in outcome units and relative to training range, interval score, point RMSE, 11 prespecified parameter/edge strata, uncertainty/error correlation and uncertainty bins.
- A fixed primary analysis: budget-16 FNO joint90 intervals, with clearly labeled provisional empirical coverage/width criteria.
- Labels/costs reported honestly: the primary workflow uses 16 training + 12 development + 39 calibration = **67 labeled settings**, plus the separate benchmark test set. Ensemble members reuse the same labels.
- Saved inference helper, Task 8/product contract, full artifacts, per-fit recovery, hashes and verified archives.

The notebook includes technical explanations and source references for ensemble uncertainty, conformal calibration, and GP posterior prediction. A successful research screen does not imply reliable arbitrary-fluid predictions or a validated adaptive stopping policy.

## Validation performed locally, 18 October 2026

**Marimo static validation:** `marimo check --strict` passed, with no notebook diagnostics. The host emitted filesystem warnings about its inaccessible personal Marimo config directory; these did not fail the check.

**10 automated tests passed**, including:

1. Initial execution stays gated; no data generation, checkpoint download or training occurs. All notebook references resolve, and critical stage dependencies explicitly connect training → sealing → calibration → final evaluation → packaging.
2. Exact finite-sample conformal order statistics, invalid inputs and insufficient-calibration guards.
3. Maximum-score joint intervals and finite scales under zero ensemble spread.
4. Independent simulated exchangeable-data coverage sanity check (a statistical implementation check, not fluid benchmark evidence).
5. Disjoint whole-setting splits, nested budgets, fixed shared supplied rows and identical calibration/test sampling distribution.
6. Rejection of exposed Task 6 seeds and invalid configurations.
7. Manufactured Taylor–Green solver check and Wilson-interval endpoints.
8. Detection of scientific-code changes and artifact tampering, while permitting explanatory Markdown changes.
9. Complete CPU integration using a deliberately tiny FNO: ten members at two budgets plus two GP workflows; per-fit resume without extra attempts; model/rule/interval seals; calibration then fresh testing; 11,200 scored interval rows; 168 global metric rows; 1,848 subgroup rows; all five figures; measured report; reloadable inference; target corruption invariance; outside-range rejection; eleven verified archives; post-evaluation weight tampering rejection. The actual Marimo downstream cells were also executed as a resumed run through plots, report, handoff and download controls.
10. The lazy Marimo download widget uses the installed API and returns the expected file bytes.

**Actual pretrained/full-size CPU smoke test:** loaded the original source checkpoint with SHA-256 `2aefbf631c8ce81f5037646b08d6e16ece9e736cbfda1b37ad4c924e261e1674`; used the complete 71,508,036-real-parameter conditioned FNO; completed one optimizer step on eight supplied settings, selected a development checkpoint, saved ~286 MB of weights, and verified reload and query-batch invariance on four label-free queries. This was a plumbing check; the development-selected checkpoint was step zero, so the smoke test makes no useful-learning claim.

**Visual check:** inspected the generated joint reliability/width figure from the tiny integration fixture. Fixture metrics intentionally have no scientific standing as Task 7 results.

**Not performed:** a full-size multi-member CUDA campaign, execution on the MoLab service, or a real customer study. Local HTML export was unavailable because the host sandbox disallows the loopback socket required by Marimo's separate kernel; actual notebook-cell execution was checked in-process instead. No remote save or deployment was attempted.

Test source is in `tests/test_task7_molab.py`. From a compatible environment, run `python -m unittest discover -s outputs/tests -p test_task7_molab.py -v` in this workspace. The integration test creates/replaces only its dedicated `work/task7-integration` fixture directory.
