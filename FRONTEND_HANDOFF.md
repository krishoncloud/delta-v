# Frontend release 4: scope and backend handoff

Historical release-4 record: the backend-pending statements below are superseded
by `CLAUDE_HANDOFF.md` and the rollout implementation added on September 12.

Only frontend, tests and documentation changed. Keep the existing Space and
production checkpoint. Do not execute the older checklist's backend steps
blindly: this checkout already serves epoch 59 from `domain_fluid.pt`, and has
a Docker Space. The live API was checked: `/predict/rollout` is absent.

## Sharing

Canonical URL: `#simulator?system=...&sample=...&field=...&frame=4`.
`tour=1` optionally enables first-use guidance. Legacy `example=1` remains
accepted as a tour flag, never as a frame index. The address bar and Share
button use the same builder from the displayed controls; changing controls
also refreshes an already-open share field. No arrays are included.

## Euler: owner-reported static results

The subsequently supplied record reports a converged 30-epoch rerun: pretrained
final loss 0.0089, scratch final loss 0.0107, and 16.3% lower error. Preserve the
reported percentage: the rounded losses cannot reproduce it exactly. Label
these measurements as training results, not live inference or per-channel L2.
The earlier short-budget comparison is superseded due to incomplete scratch
convergence. It must not appear as a numeric claim in public copy.

The gamma-selection bug yielded one distinct gamma rather than four intended.
This caveat is public. Checkpoints were not saved; the static panel must not
pretend to support Euler inference. The two arms both train model weights;
this is not the frozen-backbone learned-adapter experiment. The maturity ladder
is per lane and aspirational, not a claim of achieving Proto-LEM.

## Proposed rollout-v1 integration contract (not a deployed backend)

Each `/systems` entry may advertise:

```json
{"rollout":{"available":true,"contract":"deltav-rollout-v1","max_steps":7}}
```

Absence/false/unknown version leaves Generate disabled. This works for the three
served systems and, later, Euler when the backend actually includes it in the
served systems list with the appropriate checkpoint. No hardcoded Euler model
mapping is made in the frontend. The legacy static selector is then omitted.

`POST /predict/rollout` request (JSON metadata, not arrays):

```json
{"system":"shear_flow","sample_id":"sf_Re1e5_Sc1","n_steps":7,"revision":"CURRENT_CACHE_REVISION"}
```

Response: `application/octet-stream`, NumPy float32 C-order array shaped
`[n_steps,4,256,256]`. Canonical channel order: scalar, pressure, velocity_x,
velocity_y. Values must be **denormalized physical/code units**. The model's
internal recurrence must remain canonical/normalized; that is backend work.
Return header `X-Rollout-Meta` containing:

```json
{
  "contract":"deltav-rollout-v1",
  "system":"shear_flow",
  "sample_id":"sf_Re1e5_Sc1",
  "cache_revision":"CURRENT_CACHE_REVISION",
  "first_frame":4,
  "steps":[
    {"frame":4,"inference_seconds":1.6,"quality":null}
  ]
}
```

Supply one step entry per returned frame; this shortened example shows the
metadata shape only. `quality` may use the existing per-channel Rel. L2/RMSE/SSIM
schema, computed before display quantization, or null when reference is absent.
Do not send placeholder zeroes. The client verifies identity, revision, dtype,
shape, finite values and consecutive frames, and cancels stale results.

Generate from f4 even if the requested visible range starts at f7. The frontend
plays the selected subrange, with independent 1/2/5/10 FPS, scrubber, and always
visible error chart. Missing metrics are chart gaps. Error need not increase
monotonically; do not force scores to increase to pass a test.

The existing samples contain exactly f0–f4. They **do not** supply f5–f10 ground
truth. The frontend does not fetch additional reference files. It displays an
approximate difference only where a loaded quantized reference exists; this
image is explicitly labeled and is never used to calculate scientific scores.
Longer reference arrays, exact full-precision evaluation and extraction provenance
are backend/data preparation work still required for complete multi-step errors.

The current adapter accepts synchronous binary responses with the existing
120-second request timeout and a 32-step frontend memory guard (~32 MiB of
prediction floats). This is not a measured CPU latency cap. The backend must
advertise its measured safe cap; if jobs are necessary, agree a versioned
submit/poll/cancel contract and update the adapter before enabling it. No fake
job support or invented endpoint is shipped.

## Launch status

Working: canonical sharing, static Euler panel and two-arm explanation, public
gamma caveat, per-lane maturity framing, rollout range/player/chart frontend.
Not live: multi-step generation on any system, extended reference comparisons,
or Euler inference. These require the separate backend and saved-checkpoint work.
The public site states these limitations; do not announce them as completed.
