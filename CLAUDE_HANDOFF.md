# Delta-V handoff to Claude — September 12, 2026

Continue frontend work in https://github.com/krishoncloud/delta-v. Production:
https://krishmalik-delta-v.hf.space (Space `KrishMalik/delta-v`). Keep this
FastAPI + plain-JS/CSS deployment; do not migrate hosting or replace working APIs.

## What the user wants

Ship an honest prototype. Users choose a physical system, recorded example,
field, and generated frame range (default f4–f10). Seed f0–f3, recursively
predict f4 onward, play the generated frames at a chosen FPS, and keep the
reference/difference/error-vs-step view visible. FPS means playback speed,
not model throughput. Do not hide disappointing results.

## Implemented

- Shared canonical URL uses `system`, `sample`, `field`, `frame`; optional
  `tour=1` is separate. Legacy `example=1` only enables the tour. Share reads
  visible controls. Explicit `window.history.replaceState` avoids collision
  with the app's saved-run array named `history`. Browser-tested selection
  of horizontal velocity and frame 2, generated link, and restoration.
- Static Euler Generalization Probe explains both full-training arms:
  SEM-pretrained initialization vs random initialization, converged 30 epochs,
  losses 0.0089 vs 0.0107, owner-reported **16.3% lower error**. Do not derive
  a new percentage from rounded losses or publish the superseded early figure.
  Public caveat: only one gamma represented instead of four intended. No saved
  fine-tuned checkpoint; **Euler is not live** and must not run the joint model
  under a misleading Euler label.
- Lane-1 maturity framing: SEM → Multi-system SEM → Proto-LEM → LEM. Evidence
  is “in the direction of,” not a claim that Proto-LEM has been achieved.
  Generalizing to one unseen fluid system is not universal physics.
- Real normalized-space autoregressive backend and capability-gated frontend
  range input, player, FPS, scrubber, reference/difference panels, visible chart.
- Optional Current workplace and LinkedIn fields, non-blocking URL guidance,
  explicit public-GitHub privacy disclosure. Prepared text stays client-side.
  Exact synthetic prepared body reached GitHub issue #1, was read back and
  verified, then closed (not deleted): https://github.com/krishoncloud/delta-v/issues/1.
- Security/input safeguards and written findings in `SECURITY_REVIEW.md`.

## Backend contract — preserve it

`GET /systems` advertises per-system
`rollout: {available:true, contract:"deltav-rollout-v1", max_steps:30}`.
Absent capability means no live generation. Served systems are
`turbulent_radiative_layer_2D`, `rayleigh_benard`, `shear_flow`.

Frontend request to `POST /predict/rollout`:

```json
{"system":"shear_flow","sample_id":"rollout-test-shear_flow","n_steps":7,"revision":"GET /health cache_revision"}
```

Response is binary NPY float32 `[n_steps,4,256,256]`, denormalized code units.
Canonical order: scalar, pressure, velocity_x, velocity_y. `X-Rollout-Meta`
is JSON containing contract/system/sample_id/cache_revision/first_frame=4,
checkpoint_epoch, total inference_seconds, mean_step_seconds, and consecutive
`steps:[{frame:4,quality:{...}},...]`. Quality uses the existing per-channel
Rel. L2/RMSE/SSIM schema; null means unavailable, never zero error.

For external clients, provide `history_frames` instead of `sample_id`:
raw `[4,native_H,native_W,native_fields]` finite numeric arrays, native grids
from `/systems`. This form returns JSON `predicted_frames`, `canonical_fields`,
system/checkpoint_epoch/metadata. Exactly one seed source is required. The UI
must retain binary transport; do not serialize its arrays into URLs or JSON.

`n_steps` is a strict integer 1–30. Generate all intermediates from f4 even if
only f7–f10 will be shown. 409 means stale revision, 429 means CPU busy
(Retry-After:10), 503 means warming up, 422 invalid input. Browser cancellation
prevents stale display but does not kill CPU work already running on the server.

Core `_prepare_history` canonicalizes/resamples/normalizes only four seed frames.
`run_fno_rollout` retains normalized canonical 256² tensors throughout recurrence.
Denormalize copies only for output. Never round-trip predictions through native
resolution/order—especially Euler, whose raw energy channel is not recoverable.

## References and physical correctness

The six original samples remain five-frame training demonstrations. Three new
IDs `rollout-test-{system}` contain 11 frames: f0–f3 seed and f4–f10 reference.
The existing top timeline deliberately shows only f0–f4; rollout uses all loaded
reference frames. Backend scores use full-precision references resampled to
model resolution. The browser's difference image uses quantized reference
pixels and is explicitly approximate; never calculate published metrics from it.
After f10 there is no reference and no measured error.

Reference source revisions/trajectory/time/shape/SHA256/byte counts are recorded
in `samples/rollout-index.json`. Clips are stored separately under HF dataset
`rollout_samples/` at immutable revision `148fcb49964f35cfc6040ab80c373cc682ca5808`.
Startup downloads only these three bounded files, verifies SHA256 and exact size,
and skips failed clips while preserving the original examples. The reproducible
bounded extractor is `research/rollout_references.py`. Source files come from
official test splits, with owner-supplied checkpoint/notebook lineage; this is
a small probe, not a comprehensive benchmark or a newly trained model.

Never touch the artifact repository's 92 GB `slices/` directory. Never retrain
or modify the production checkpoint. Epoch 59 `checkpoints/domain_fluid.pt`
is intentional; the older `_best` file is not automatically better.

## Measured runtime and safety

Live shear-flow 30-step response: shape `[30,4,256,256]`, all finite,
54.72 seconds computation (1.824 seconds/step), 85.03 seconds including transfer.
The first step agrees with the earlier one-step metrics. The 120-second browser
timeout leaves limited margin for slower connections/load. Default is seven
steps; do not promise real-time generation. Keep 30 as a hard bound, with a clear
long-request warning. Async submit/poll could be justified if sustained traffic
or measurements exceed that envelope; no fake job system is implemented.

Unit coverage checks normalized recurrence (including Euler preprocessing),
single seed conversion, shapes, strict caps, unknown systems, revisions, missing
reference gaps, body limits, sharing, caching and cancellation. Error is not
required to increase monotonically: genuine fluctuations are valid. Never
alter a curve to make it look like accumulating error.

Local checks: 14 Python tests and 12 JavaScript tests passed. Run
`python -m pytest -q` and `node --test tests/frontend.test.cjs` from the repo.
The dependency audit output is `dependency-audit.json` (71 resolved packages,
zero reported advisories; see the production-inventory caveat in the review).

Live seven-step checks passed for all three official-test clips: every response
was finite float32 `[7,4,256,256]`, every f4–f10 step had full-precision reference
metrics, and each first frame matched `/predict/field` within 1e-6 tolerance.
Results are recorded in `rollout-verification.json`; rerun with
`python research/verify_live_rollout.py --output rollout-verification.json`.

| System | Compute for 7 steps | Scalar Rel. L2 f4 → f10 |
|---|---:|---:|
| Cooling-driven turbulence | 13.45 s | 9.21% → 36.86% |
| Heat-driven convection | 13.47 s | 9.05% → 22.39% |
| Shear-driven mixing | 13.19 s | 3.09% → 36.63% |

These are measured errors on three small clips, not universal rollout accuracy.
The final server check reported epoch 59, nine cached predictions/nine samples,
and 43.76 seconds startup including artifact/model loading and precomputation.

## Frontend follow-through for Claude

1. Make the official-test f0–f10 example easy to find beside Generate; keep
   original training examples clearly distinct. Preserve accessible controls.
2. Polish the generated-vs-recorded playback hierarchy, loading/cold-start
   states, narrow-screen layout and latency wording. Keep error chart visible.
3. Preserve shared-selection state and pending-request cancellation while
   changing routes or samples. Share links currently cover the one-step
   selection; adding rollout range/frame sharing needs a versioned extension.
4. Do not present Euler generation as ready until the owner reruns the gamma
   selection fix, saves checkpoints and supplies a real serving mapping.
5. Private feedback storage, a Linux deployment lockfile/SBOM audit, stricter
   CORS allowlist and traffic rate limits remain product/infra follow-ups.
   Current feedback is an optional public GitHub issue, not a private inbox.

Read `SECURITY_REVIEW.md` for explicit CORS, secrets/history, dependency, input,
rate-limit, XSS and feedback-storage findings. `FRONTEND_HANDOFF.md` is historical;
its backend-pending claims have been superseded by this release.
