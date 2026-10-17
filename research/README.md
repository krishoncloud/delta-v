# Reproducible evaluation and transfer work

## Next feature release: new problems within the fluid lane

See the [15-task release plan and new-problem acceptance criteria](V2_RELEASE_PLAN.md).
Task 4 supplies the reference data and GP benchmark; Tasks 5–6 must build and test
pretrained adaptation on a new customer-style fluid problem before the product
claims that advantage. Task 15 must complete the public/private separation before
the next feature launch; its implementation is currently deferred.

## Task 4: outcome tables and GP baseline

Use [the native MoLab notebook](molab_task4.py) for the scalar-outcome benchmark.
It pins dependencies, embeds the small verified tables, and saves each completed
run to Hugging Face with remote hash verification. CPU compute is sufficient.
See [task 4 instructions and committed results](task4/README.md) for setup,
the full artifact inventory, and the measured pilot's limits.

These offline tools never download `slices/` and do not modify the served model.
The supplied notebook was inspected as source, not executed. Its ungated cleanup
cells delete local/remote checkpoints and Euler slices; do not run those cells
as part of reproduction.

## Findings from the supplied notebook

- `open_well` uses only `well_split_name="train"` (lines 300–313).
- Step 8 estimates normalization from sampled windows before splitting (701–718).
- Step 12 calls `random_split` on concatenated sliding-window datasets with
  seed 42, 15% validation (1128–1135). It does not isolate trajectories.
- Step 14 implements full fine-tuning versus scratch on Euler, using another
  random-window 80/20 split with seed 123 (1411–1425). No learned adapter arm
  or independent test set exists there. The export contains code, not results.
- Euler training uses fresh models `mA`/`mB`; the joint model training family
  excludes Euler. No checkpoint writes occur in the inspected Euler arm.
- The joint checkpoint does not store the split membership or source revisions.

This establishes the intended train-only pipeline, not cryptographic proof of
checkpoint lineage. Do not call the old validation loss trajectory-held-out.

## Small official-test probe

`source_probe.py` declares six fixed windows from pinned official `data/test/`
files, extracts them through exact HTTP Range requests (256 KB blocks, 96 MB
maximum per source), and asks the existing binary field API for predictions.
It computes float-field metrics locally with the same preprocessing. All
source revisions, file names, trajectory indices, start indices, window
hashes, checkpoint artifact revision and notebook hash are recorded.

The second-file selection replaces an unavailable second trajectory in the
radiative dataset. It is based on source dimensions, not model scores. Do not
tune on this small test set. It cannot establish broad generalization.

## New trajectory-level split

Prepare an inventory of local `.npy` windows (five raw snapshots, native field
order) with fields `system`, `source_revision`, `source_file`, `trajectory`,
`start_index`, `path`, and `sha256`. All windows of one trajectory share a key.
Run `python research/split.py inventory.json manifest.json --seed 42` to assign
approximately 70/15/15 train/validation/test, with at least one trajectory in
each split per system. The output file must not exist. Do this before training;
it cannot create an unseen split retrospectively for epoch 59.

For evaluation manifests also provide:

```json
{
  "checkpoint": {
    "sha256": "EXACT_CHECKPOINT_HASH",
    "exposure_complete": false,
    "provenance_note": "Cite the actual training run and membership records",
    "exposed_trajectories": []
  },
  "statistics": {
    "SYSTEM_NAME": {
      "path": "LOCAL_STATS_FILE.npz",
      "sha256": "EXACT_STATS_HASH",
      "provenance_verified": false,
      "fit_trajectories": []
    }
  },
  "windows": []
}
```

Placeholder manifests intentionally fail validation. Exposure must include
training, validation/model-selection, and normalization data. Trajectory keys
are compact JSON arrays `[system, source_revision, source_file, trajectory]`.
Never set provenance flags without checking the underlying records.

`python research/evaluate.py --manifest manifest.json --checkpoint model.pt
--output results.json` verifies hashes and disjointness, evaluates all three
systems, and reports each window plus means with equal trajectory weighting.

## Three-arm Euler experiment

Prepare the four-gamma dataset with bounded HTTP ranges:

`python research/prepare_euler.py --output-dir NEW_DATA_DIRECTORY
--notebook DeltaV_FluidDomain_molab_v2.py --checkpoint domain_fluid.pt`

The default selects 16 trajectories from each of gamma 1.13, 1.22, 1.33 and
1.365, then assigns fixed 8/4/4 train/validation/test trajectories per gamma.
It stores four five-frame windows per trajectory (starts 0, 32, 64 and 96),
giving exactly 32 training trajectories and coverage for every requested
budget. Each source is read through exact ranges with a 160 MiB cap per
trajectory. No 21 GB source file and never the 92 GB `slices/` tree is
downloaded. Expect about 6.4 GB of range traffic and local raw windows.

`python research/experiment.py --manifest manifest.json --checkpoint model.pt
--output-dir NEW_DIRECTORY --device cuda --trajectory-budgets 1 2 4 8 16 32
--seeds 42 --steps 1000 --validation-interval 100` compares scratch, full fine-tuning,
and a frozen FNO with identity-initialized learned 1×1 input/output projections
(292 trainable real parameters). The production architecture is untouched.

Trajectory budgets are nested and selected by stable trajectory hashes. They
remain identical across random seeds, so seed variance is not confounded with
different training data. Use one seed for the pilot, then `--seeds 42 123 2026`.
The same validation data and optimization-step budget are used for all arms;
the best validation checkpoint is selected. Pilot runs report validation metrics
by default. After fixing the protocol, use `--evaluate-split test` once per final
arm, budget, and seed. The script reports actual and requested trajectory counts,
seed, best step, trainable and
total real parameter counts (complex parameters count twice), wall time and
per-channel metrics. Target normalization is fitted separately on each budget's
training windows only and saved with a hash; all three arms use those same
statistics. Validation and test windows never contribute. Default settings are a baseline protocol, not an optimized
or completed experiment. Repeat across seeds before making a transfer claim.

Actual adapter experiments require prepared Euler train/validation/test data
and a suitable training runtime. The supplied notebook assumes a GPU; this
repository does not provision or purchase compute.

## Task 5: parameter-conditioned FNO notebook

Import [molab_task5.py](molab_task5.py) directly into MoLab. This single file
contains all 14 stages of the scoped Task 5 development workflow, from study
validation through actual FNO adaptation and verified local artifact downloads.
See [Task 5 instructions and limits](task5/README.md) and
[validation evidence](task5/validation.md). The default exact vortex is an
implementation test; it does not establish customer-run savings or FNO superiority.
New customer artifacts and adapted weights are not uploaded automatically.
