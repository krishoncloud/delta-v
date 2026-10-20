# Task 6 — completed new-fluid-study benchmark

The supplied **Task 6b MoLab GPU results** are preserved in
[`results/20261017T173214Z-task6b/`](results/20261017T173214Z-task6b/).
Start with the [original measured report](results/20261017T173214Z-task6b/report.md),
[runs-to-tolerance table](results/20261017T173214Z-task6b/runs_to_tolerance.csv),
[paired comparisons](results/20261017T173214Z-task6b/paired_comparisons.csv), and
[data-efficiency figure](results/20261017T173214Z-task6b/data_efficiency.png).
The exact executed notebook
is retained unchanged alongside the results; this publication does not rerun training.

## Measured result and scope

All 72 required fits completed: six methods × four nested supplied-setting
budgets (8/16/24/32) × three seeds. The generated target is a three-control,
fixed-geometry periodic incompressible fluid study. Outcomes are kinetic energy
and pressure RMS. This run used 3,000 optimizer steps per neural fit on an NVIDIA
RTX PRO 6000 Blackwell Server Edition.

- Pretrained full tuning reached the locked provisional accuracy target at **16
  supplied settings**; scalar GP and scratch FNO reached it at **24**. This is
  eight fewer supplied training settings, or 33% fewer than 24, on this study's
  tested budget grid. It is retrospective data efficiency, not demonstrated
  prospective solver savings.
- Pretrained/GP aggregate error ratios were approximately **0.32, 0.86, 1.76,
  and 2.86** at budgets 8, 16, 24, and 32. GP therefore overtook the pretrained
  model at the larger budgets. A blanket GP-superiority claim for the FNO is
  unsupported.
- The locked sparse-budget rule for **pretraining versus scratch was not met**.
  Ratios at 8 and 16 were about 0.98 and 0.94, with inconclusive descriptive
  paired intervals. At 24 and 32 the ratios favored pretraining (about 0.74 and
  0.68); these do not replace the prespecified primary sparse-budget decision.
- The frozen spectral-block arm did not reach the target tolerance. In this
  Task 6b arm, the conditioner plus lifting/projection layers trained while the
  spectral blocks remained frozen; it is not a wholly frozen backbone.

The FNO receives completed training fields in addition to scalar outcomes; the
GP receives controls and scalar labels. This comparison does not isolate field
information from architectural effects. POD+GP was not run. There is one
synthetic target family and three seeds, not independent customer validations.
Source-exposure limitations and negative results remain in the original report.

There were 32 training-pool, 12 development, and 24 confirmation settings in the
full benchmark, plus disclosed numerical-audit work. The **16-setting** figure
counts supplied training settings, not all simulations needed for development
and evaluation. Tasks 7–9 address calibrated uncertainty, prospective acquisition
and a real solver/customer case respectively. Source-diversity/pretraining-scaling
follow-up Task 6c-B is deferred; it is not represented as a completed GPU run.
Task 15 private-storage separation remains required before launch.

## Preserved evidence

All **1,126 files supplied in `TASK_6_result.zip`** are extracted unchanged:
protocol, original notebook, source lineage, environment/session, design and
reference fields, per-fit contracts and receipts, development predictions,
confirmation predictions, numerical audits, warnings, cost ledgers, metrics,
figures and report. No new checkpoint or customer data was published.

The input ZIP passed its CRC check. Its SHA-256 is:

```text
0f197d13d49526e8528ecd9d6010c0c27c2d45f8d96372b833efdee76a84b2eb
```

The protocol hash was recomputed and matches:

```text
7fb9154efb8a8c0eae581c92d2c024d918c0548cdb1210ba64fa33ffec7b7ee2
```

[Publication verification](publication_verification.json) records the checks and
known omissions. [Published-file manifest](published_files_manifest.json) hashes
every one of the 1,126 supplied files. There were 1,852 successful original
manifest checks including repeated references, covering 979 distinct files;
manifest files excluded by the original manifest generator are also covered by
the new publication manifest.

**Weight limitation:** the original evidence ZIP excludes the 36 selected neural
weight files and two source-cache files referenced by its root manifest. Those
files were not supplied and are not in this publication. Original manifests are
preserved rather than altered to hide the omissions. This is a complete copy of
the supplied research-evidence archive, not a complete backup of all adapted
models. Reloading an adapted neural model requires the separately downloaded
weight archives; the source checkpoint identity alone cannot reconstruct it.
