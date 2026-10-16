# Task 4: scalar-outcome dataset and GP baseline

[Open the native notebook in MoLab](https://molab.marimo.io/github/krishoncloud/delta-v/blob/research/task4-outcomes/research/molab_task4.py)

The supported MoLab entry point is [`../molab_task4.py`](../molab_task4.py).
It is a self-contained native marimo notebook with PEP 723 dependency pins,
embedded hash-verified public HF tables, explicit execution, and verified HF saves.
CPU compute is sufficient. No FNO checkpoint or large simulation fields are downloaded.

## Run in MoLab

1. Open the `.py` notebook using the link above, or import its GitHub URL in MoLab.
2. Let the package environment install. The notebook declares Python 3.11–3.13
   and exact versions for marimo, NumPy, SciPy, pandas, scikit-learn, matplotlib,
   joblib, threadpoolctl, huggingface-hub, and python-dotenv. If packages are missing, use the
   package panel's manifest; the notebook does not invoke pip inside reactive cells.
3. Add `HF_TOKEN` in MoLab Secrets, with write access to the existing dataset
   `KrishMalik/deltav-fluid`. Alternatively save `HF_TOKEN=your_token` in `.env`
   beside the notebook. The notebook reads it on Prepare and save/retry, without
   printing it or copying `.env` into artifacts. Existing environment secrets take
   precedence. Never paste the token into notebook source.
4. Review `task4_config` in step 2 and click **2 · Prepare experiment**. Keep automatic
   HF saving enabled. Inspect the data audit, source audit, split plan and 128 preflight
   checks in steps 3–8. Preparation stops before benchmark training.
5. Click **9 · Run benchmark**. Steps 10–13 display the learning curves, uncertainty,
   worked example, matched comparisons, reusable-model checks and final report.
6. In step 14, wait for **Saved and verified on Hugging Face** and open the exact run link.
   A failed upload leaves the completed local artifacts and ZIP available.
   **Retry save** retries persistence without repeating the benchmark.

Future runs are uploaded to:

```text
KrishMalik/deltav-fluid
  research/task4-outcomes/v1/runs/<unique-run-id>/
```

Each run is uploaded in one HF commit, with a parent-revision guard. The save routine
checks every file against the local manifest before uploading, then verifies every
remote file's size and Git-blob or LFS hash at the returned immutable revision.
Retries verify existing content and never overwrite a different or partial run.
A local receipt records the verified commit URL. A receipt is not an experiment
artifact and is written alongside the run directory after verification.

The ZIP is a downloadable packaging of the same files; the HF repository stores
the individual files so they can be inspected without downloading an archive.
The notebook's GitHub synchronization does **not** itself synchronize output folders.

For local use:

```bash
uvx marimo edit --sandbox research/molab_task4.py
```

Uncheck automatic HF saving only if you intentionally want a local-only run.

## Notebook structure

The native notebook has 14 explained steps across 36 reactive cells, following
`research/molab_track_a.py` on the Euler research branch. Each scientific stage
executes in its own cell; there is no monolithic `run_task4()` wrapper.

| Step | Action and visible evidence |
|---|---|
| 1–2 | Import pinned dependencies, choose seeds/restarts/tolerances, prepare a unique run |
| 3–4 | Verify embedded source hashes, validate schema, show setting counts and input ranges |
| 5 | Audit official source filenames for additional parameter settings |
| 6 | Lock the protocol and show test sizes, training pools, anchors and budgets |
| 7–8 | Explain GP/mean/linear models and run leakage, scaling and determinism checks |
| 9 | Explicitly launch the benchmark and retain all fits and warnings |
| 10 | Display learning curves and uncertainty coverage alongside interval widths |
| 11 | Show supplied data, query predictions and hidden-reference comparison |
| 12 | Compare methods on matched evaluations; optionally evaluate predeclared tolerances |
| 13 | Save/reload GP bundles, check full execution, produce report and artifact manifest |
| 14 | Download all artifacts and verify automatic HF persistence, with retry |

Click Prepare again and then Run benchmark to start a fresh run. Edits to the
configuration invalidate downstream results. Do not edit execution cells mid-run.
The check lists are copied between stages so a rerun does not accumulate duplicate
correctness records. Original reference evidence remains immutable.

## Current committed evidence

[`results/20261016T033030_282756Z/`](results/20261016T033030_282756Z/)
contains the full verified native-notebook reference run. All 43 manifest-listed
files, plus the manifest itself, are committed. The native run's metrics, predictions,
and summary match the original Jupyter execution exactly.

| Files | What they preserve |
|---|---|
| `data/*.csv`, `data/schema.json` | Original rows, grouped settings, inputs/outcomes, provenance gaps, example inputs and reference outcomes |
| `sources/` | Exact NPZs, pinned official source inventories, and the notebook source that generated the run |
| `protocol.json`, `environment.json` | Seeds, splits, training order, budgets, model settings, versions |
| `metrics.csv`, `predictions.csv`, `summary.csv` | Full-precision per-split errors, held-out predictions, uncertainty, and summaries |
| `fit_details.json` | Training normalization, kernel fits, optimizer warnings, timing, row membership |
| `paired_comparisons.csv`, `comparison_summary.csv` | GP versus mean and linear baselines on matched data |
| `data_audit.csv`, `source_inventory.csv`, `expansion_audit.json` | Data checks and source-expansion feasibility |
| `example_predictions.csv`, `target_analysis.json` | Predeclared measured example and optional tolerance analysis |
| `models/*.joblib`, `models/index.json` | Three all-setting GP models plus the worked-example GP, including preprocessing and training arrays |
| `figures/`, `REPORT.md` | Learning curves, uncertainty, worked example, and written interpretation |
| `checks.json`, `completion.json`, `artifact_manifest.json` | Correctness outcomes, completion scope, and SHA-256 integrity manifest |

The four serialized model bundles are for reuse. Benchmark fits at every split/budget
are reproducible from the saved data, protocol, normalization, and code; they are not
all separately serialized. The all-setting models are fitted after evaluation and are
not used to inflate held-out scores. Only load joblib files from trusted sources.

[`DeltaV_Task4.ipynb`](DeltaV_Task4.ipynb) retains the original executed Jupyter
reference, including plots. Use the native `.py` notebook for MoLab and HF persistence.

## Validation

- `marimo check research/molab_task4.py` passes.
- Native execution: 936 benchmark model fits; 9,954 held-out scalar predictions;
  140 embedded correctness checks, including saved-model prediction reproduction.
- Two stage integration tests verify the preparation gate, exact default-result
  equality with the reference, separate check lists, source snapshot and all 43 artifact hashes.
- Nine offline persistence/credential tests cover Git/LFS verification, retry idempotency,
  local/remote corruption, untracked files, partial runs, upload failure, and missing tokens.
- No authenticated HF upload or hosted MoLab session was performed during local
  verification. The first MoLab run must show the remote verification status before
  its results are considered durably saved.

```bash
python tests/test_task4_molab.py
```

The staged revision was also exercised through a local browser: prepare, pause,
benchmark, plot/report rendering and local-only download availability. See
[`staged_validation.json`](staged_validation.json) for the revision verification record.

## Scientific scope

There are 9 cooling-time, 35 Rayleigh–Bénard, and 28 shear-flow settings. Tables
contain averaged outcomes at each setting, not individual solver-run records.
The audit preserves incomplete physical-unit and trajectory-lineage information.
The historical `mass_flux` column is scalar-times-vertical-velocity; its physical
meaning differs between density, buoyancy, and tracer systems.

The benchmark includes fixed nested budgets, common training/test settings for all
three methods, training-only fitting, interpolation, mixed holdout, boundary
extrapolation, and coverage/width evaluation. Results and warnings are retained
whether or not the GP wins. No customer tolerance is invented. These results do not
establish transfer-learning advantages or solver-run savings.

The official-source filename audit finds no extra parameter settings at the inspected
revisions. Raw fields could improve aggregation/trajectory provenance and variability
estimates, but that is separate extraction work.

References: [MoLab workflow](https://docs.marimo.io/guides/molab/),
[marimo dependency manifests](https://docs.marimo.io/guides/package_management/sandboxes/).
