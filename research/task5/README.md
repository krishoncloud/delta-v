# Task 5 — parameter-conditioned FNO development notebook

Import **[molab_task5.py](../molab_task5.py)** directly into marimo MoLab. It is a
single self-contained notebook with pinned dependencies, 14 explained stages,
three explicit execution buttons, reusable offline functions and local artifact
downloads. It does not fetch executable code from another repository.

## Run it

1. Choose a CUDA GPU in MoLab. Start with at least 8 GB of GPU memory; the actual
   first step establishes whether the runtime fits. Package installation and the
   858 MB source checkpoint download are separate from training time.
2. Review configuration, then click **Prepare study**. The built-in study needs
   no external dataset or token. Inspect the data contract and split tables.
3. Click **Load pretrained FNO**. Checkpoint identity is pinned and its hash is
   verified before restricted loading and strict state-dictionary matching.
4. Click **Train Task 5**. Default GPU budget: 200 steps, seed 42, 900 seconds.
   CPU mode is a short integration smoke test capped at ten steps.
5. Inspect development errors, baseline comparisons, query checks and completion
   gates. Download the full ZIP and the smaller evidence ZIP before leaving MoLab.

All source and helper functions are inside the notebook. No CLI module, clone,
secret, backend service or solver installation is needed for the built-in case.
For local use: `uv run research/molab_task5.py` or
`uv run --with marimo marimo edit --sandbox research/molab_task5.py`.

## What it actually implements

Known initial fields + named knobs + coordinates → new 1×1 conditioner → the
actual pretrained FNO → a fixed-time field → physically defined scalar outcomes.
The whole backbone and conditioner are fine-tuned. Inference needs no hidden
future frames or output labels. The model, normalization, parameter order,
units, domain, observation time, output definitions and lineage are saved together.

The narrow supported contract is fixed geometry, a 64×64 regular 2D grid,
channels `[density, pressure, velocity_x, velocity_y]`, known initial conditions,
and one fixed observation time. Source weights were used at 256×256; the notebook
explicitly records this resolution change. Supported readouts are specific
kinetic energy, pressure RMS and a declared pressure difference between two probes.
There is no automatic unit conversion or arbitrary solver-format import.

The default controlled Taylor–Green study has 16 training settings and 9
development settings, with 16 further control combinations reserved without
generating their reference fields or outcomes. It provides exact physical checks
and a bounded implementation test. Its two default outcomes are mathematically
equal, so they are not independent demonstrations of accuracy. This easy reference
case is not evidence of economic value or superiority on customer CFD.

The `customer_bundle` option accepts a directory containing the documented JSON,
CSV and numeric NPY files. Use the generated example as the schema template. The
notebook validates hashes, shapes, units, field-derived outcomes, whole-setting
splits and interpolation support. The customer must truthfully describe initial
conditions and source exposure; file hashes cannot prove these scientific facts.

## Reading the result

`READY FOR TASK 6` requires improved training-field and development losses,
better-than-mean development outcome error on at least one outcome, and successful
reload/conditioning/query checks. Otherwise the report says `NEEDS WORK`, retains
the evidence and does not claim completion. Development selects the checkpoint;
these scores are exploratory, not untouched test results.

GP, linear and mean use the same run IDs and scalar labels. FNO has additional
field supervision. Only a subsequent comparison with identical-input scratch
training can establish pretraining's value. Task 6 must audit a harder new fluid
study, lock budgets/tolerances, and use fresh confirmatory references. Tasks 7/8
cover uncertainty and next-run selection; Tasks 9/10 cover solver/API integration.

## Storage and privacy

The notebook has no remote-write operation. It downloads only the public pinned
source checkpoint. The full verified ZIP contains the model, study arrays,
predictions, traces, figures, report, checks, environment, exact notebook source
and artifact hashes. The evidence ZIP omits model weights and supplied field
arrays; it cannot independently reload the model. New training runs do not commit
artifacts to GitHub or upload them to the public HF dataset.

Task 15 remains paused and is still required before the next feature launch.
The publication of this research notebook is explicitly requested; it does not
make future customer data or adapted weights public.

## Checks

`python -m unittest discover -s tests -p test_task5_molab.py -v`

The integration test uses a tiny FNO **only through test overrides**, to exercise
every notebook stage offline. The notebook has no tiny-model UI or fallback.
See [validation.md](validation.md) for separate checks with the actual source FNO.
