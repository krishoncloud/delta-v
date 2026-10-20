# Task 8 MoLab notebook — run guide and local validation

Notebook: **DeltaV_Task8_MoLab.py**, a self-contained Marimo notebook with 57 cells and 18 numbered sections. It embeds the verified Task 7 model/solver path and the complete Task 8 acquisition, stopping, scoring and delivery workflow. It does not import another notebook or clone a repository.

## Faster complete default experiment

| Item | Default |
|---|---|
| Workflows | FNO active; FNO LHS; GP active; GP random; GP Sobol |
| Paired repetitions | 3, with fresh pools/development/calibration/probe/test settings |
| Initial training runs | 8 box-corner anchors |
| Acquired batch size | 8 |
| Training budgets | 8, 16, 24, 32 |
| Neural ensemble | 3 members per FNO workflow/round |
| Neural training | 3,000 optimizer steps, full adaptation from the original source each round |
| Total planned neural fits | 72 |
| GP fitting workflows | 36, each selecting kernels separately per outcome |
| Candidate pool | 512 unlabeled feasible settings per repetition |
| Development/calibration | 12/39 labels, charged to every workflow |
| Readiness probes | 128 unlabeled settings |
| Confirmation | 200 fresh numerical cases per repetition; opened after all decisions are sealed |
| Estimated runtime | Approximately 1.5–2.5 hours on the prior Blackwell GPU; not a measured Task 8 runtime |
| Recommended free disk | About 16 GB for defaults; calculated from actual model tensor size |

The initial customer-style state costs **8 + 12 + 39 = 59 labeled settings**. The training ceiling costs **32 + 12 + 39 = 83**. Test cases and numerical audits are additional benchmark evaluation work. Shared solver caching does not make labels free for any competing workflow.

The FNO/LHS comparison isolates selection effects using the same model/information/training recipe. The GP comparisons evaluate complete alternative workflows; FNO additionally receives supplied fields and incurs ensemble training costs. No POD/field-feature GP arm is present.

The faster defaults remove 66% of the original neural fits while preserving all three repetitions, all five workflows and 3,000 optimizer steps. Eight-run batches provide coarser stopping/savings resolution; three-member uncertainty is noisier and must be recalibrated and evaluated. The minimum savings screen is now eight labels, matching the batch. No Task 7 coverage result is assumed to transfer.

The original protocol is preserved in `outputs/task8_original_full/`. To select it in the new notebook before beginning: acquisition batch 4, member seeds `1103,2207,3301,4409,5501`, ceiling 32. Scientific configuration/code changes require a new run directory; existing results are not migrated or relabeled.

## Run on MoLab

1. Import `DeltaV_Task8_MoLab.py` and choose CUDA. Read the introduction and goal before submitting the configuration form.
2. Apply the full-run configuration, with an output directory on adequate storage. Use the same directory/configuration to resume; choose a fresh master seed and directory after inspecting test outcomes from an earlier experiment.
3. Click **Prepare / resume fresh Task 8 studies**. Inspect the numerical audit, data-role counts, resource plan and automatic scientific protocol lock.
4. Click **Load verified original FNO source**. The original public checkpoint is hash-pinned; no HF token is required.
5. Click **Run / resume FULL Task 8 closed-loop benchmark**. There is no mandatory pilot. The default is all 72 neural fits and 36 GP fitting workflows. Progress and per-fit/round receipts are saved continuously.
6. Once all trajectories and first stopping proposals are sealed, click **Reveal sealed test answers and evaluate Task 8**. Read the stopping, savings, subgroup and cost tables, figures and measured report.
7. Click **Verify and package Task 8 evidence and retained models**. Download the evidence ZIP, delivery receipt and retained weight files before ending MoLab.
8. To restore models after downloading: extract the evidence ZIP to a run folder, put the individual weight downloads in a downloads folder, and run its `restore_weights.py --run-dir <run-folder> --downloads <downloads-folder>`. It validates hashes, reconstructs original paths and refuses to overwrite mismatched files.

A new result is not established by rerunning already-inspected confirmation cases. A failed or time-capped member cannot seal the experiment. Completed fits/rounds resume from receipts; failed fits restart from their original seed. Hard runtime termination can lose the current fit's unrecorded elapsed time. The source cache persists independently of run-button pulses.

## Stopping, savings and limits

The engineering goal is both outcome errors within 0.015 for at least 90% of fresh test settings, with maximum error/tolerance ratio at most 2. These are provisional research thresholds, not customer safety limits.

The stop proposal uses calibrated joint90 interval half-widths on label-free probes. It does not use test truths. Later independent scoring identifies premature stops. First oracle-qualified budget is a separate retrospective diagnostic and cannot replace a bad first stopping proposal.

The primary descriptive research screen compares FNO active with GP active: every paired repetition must have a correct first proposed stop for both workflows, savings must be positive in each repetition, and median total-label savings must be at least one eight-run batch. Other baselines and the FNO/LHS attribution comparison remain visible even if they outperform the primary workflow.

After its first proposed stop, each research trajectory continues to the ceiling to measure curves/regressions. Those rounds are marked shadow continuation. Customer-prefix costs stop at the proposed stop, while the actual benchmark ledger includes shadow work. This is a controlled sequential-solver benchmark with an independently evaluated stopping heuristic, not a real customer deployment or a guaranteed-safe stopping policy.

Calibration labels never enter training or acquisition. Reusing calibration over the fixed acquisition path does not supply an optional-stopping, conditional-region, or adaptive-query coverage guarantee. Round/stopped-model coverage and high-forcing/low-initial-mix failures are measured explicitly.

Standalone prefix runtime estimates add all successful/failed fitting attempts, measured non-fit loop/file overhead, and the recorded solver time for each unique consumed label. Shared-cache input-access time is excluded to avoid confusing cached execution with a standalone solver cost. Source load and unknown source-pretraining/export costs are separate; these estimates are not independently measured customer wall clocks. Breakeven expensive-solver times are scenario calculations, not money saved.

## Storage and artifacts

Retain the first proposed-stop and latest FNO round for each workflow/repetition. Superseded intermediate weights are pruned only inside this run's owned workflow folders, only after a later round is committed, and with checksum/replacement receipts. All intermediate predictions, acquired IDs, seeds, contracts, learning traces, metrics and reconstruction recipes remain. Reconstructing an intermediate neural model requires retraining; bitwise hardware-independent reproducibility is not promised.

The evidence ZIP excludes the original source tensor cache and retained neural weights. Individual retained weights are downloaded separately to avoid another approximately 10 GB copy. `weight_inventory.json` records their original relative paths, download filenames, sizes and hashes. `artifact_manifest.json` covers retained run artifacts. Earlier pruned weights are documented by their receipts, not falsely represented as supplied models.

The notebook performs no GitHub/HF upload, deployment, paid compute purchase or private-repository migration. Task 15's private storage is still required before launch. The test uses one fixed periodic fluid family; arbitrary geometry/3D/customer compatibility and a real solver study remain outside this task. Task 6c-B stays deferred.

## Local validation performed on 18 October 2026

**Strict Marimo check passed** with no notebook diagnostics. The host emitted filesystem warnings about its inaccessible personal Marimo config directory; these do not affect the notebook check.

**All 13 automated tests passed**, covering:

1. Initial execution is gated, all notebook references resolve, and critical stage dependencies explicitly connect campaign → trajectory sealing → confirmation → report → packaging.
2. The original 210-fit/five-member protocol remains configurable; the faster public defaults plan 72 fits, 36 GP workflows and 60 workflow rounds. Honest 59/83 total-label accounting is unchanged.
3. Disjoint whole-setting roles, fresh paired repetitions and shared corner anchors.
4. Deterministic feasible unique random/Sobol/LHS acquisition sequences.
5. Active selection uses only declared label-free inputs, responds to uncertainty and avoids repeated runs.
6. Exact conformal order statistics and a readiness proposal that remains distinct from actual hidden-test correctness.
7. Premature stops and censored/non-stopping baselines cannot manufacture valid savings.
8. Manifest/scientific-fingerprint tampering detection, Markdown-only edit tolerance, and unsafe restore-path rejection.
9. A complete miniature CPU benchmark across all five workflows and two acquisition rounds: **8 tiny-FNO fits + 6 GP workflows, including acquisition from 8 to 16 runs**, real solver calls for selected cases, acquisition/refit ID matching, fit/solver-cache resume, safe intermediate pruning, sealed confirmation, **400 scored rows and 110 subgroup rows**, all four figures, report/costs/handoff, saved query-only inference, hidden-test-label corruption invariance, outside-range rejection, evidence packaging and a successful generated standalone weight-restoration script. The actual downstream Marimo cells were also executed as a completed resumed run through figures, report and downloads.
10. Positive paired-savings arithmetic includes calibration/development overhead, while one premature FNO stop invalidates the consistent-win screen.
11. Retention preserves both first-stop and latest models, prunes only a superseded intermediate, validates pruning receipts, and explains missing retained-weight restoration.
12. A time-capped fit is recorded as failed and cannot produce a completed-fit receipt.
13. Faster defaults: 72 neural fits, 36 GP workflows, all five policies select eight unique feasible runs, budget alignment and retained-model/storage accounting.

The miniature integration used one repetition, two ensemble members, two optimizer steps, a small one-layer FNO, a 64-case candidate pool and 20 confirmation cases. It explicitly tests machinery, not scientific performance or GPU capacity. The public form defaults remain the complete faster protocol; the two-member fixture is only for machinery validation.

**Original-source, full-size CPU smoke check (rerun on this faster notebook):** the 858 MB source checkpoint matched SHA-256 `2aefbf631c8ce81f5037646b08d6e16ece9e736cbfda1b37ad4c924e261e1674`. The actual 71,508,036-real-parameter conditioned FNO completed one optimizer step, saved approximately 286 MB of weights and passed saved-model reload and query-batch invariance checks on four label-free queries. Development selected step zero in this smoke run; it establishes plumbing, not useful learning or Task 8 savings.

**Visual inspection:** inspected the stopping/readiness figure from the deliberately unsuccessful miniature fixture. These are test artifacts, not reportable Task 8 research results.

**Not performed:** the full 72-fit CUDA campaign, execution on the MoLab service, a customer/OpenFOAM study, remote upload or deployment. There are no measured Task 8 wins to claim until the full experiment executes.

Run the automated checks in a compatible environment with:

```bash
python -m unittest discover -s outputs/tests -p test_task8_molab.py -v
marimo check --strict outputs/DeltaV_Task8_MoLab.py
```

The integration test creates/replaces only its dedicated `work/task8-integration` fixture directory and adjacent evidence archive.

The additional faster-profile check verifies all five acquisition policies request eight unique feasible runs, the 72/36 job counts, 36 retained-model ceiling, storage estimate and rejection of misaligned budgets. The complete miniature integration also exercises the eight-run acquisition/refit path.
