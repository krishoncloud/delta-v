# Reproducible evaluation and transfer work

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

`python research/experiment.py --manifest manifest.json --checkpoint model.pt
--output-dir NEW_DIRECTORY --device cuda` compares scratch, full fine-tuning,
and a frozen FNO with identity-initialized learned 1×1 input/output projections
(292 trainable real parameters). The production architecture is untouched.

Fractions are nested, selected by stable trajectory hashes with one seed.
The same validation data and epoch budget are used for all arms; the best
validation checkpoint is selected before test evaluation. The script reports
actual trajectory counts, requested fraction, seed, best epoch, trainable and
total real parameter counts (complex parameters count twice), wall time and
per-channel metrics. Target normalization is fitted separately on each fraction's
training windows only and saved with a hash; all three arms use those same
statistics. Validation and test windows never contribute. Default settings are a baseline protocol, not an optimized
or completed experiment. Repeat across seeds before making a transfer claim.

Actual adapter experiments require prepared Euler train/validation/test data
and a suitable training runtime. The supplied notebook assumes a GPU; this
repository does not provision or purchase compute.
