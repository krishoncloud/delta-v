---
title: Delta-V Inference API
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Delta-V Inference API

Physics-simulation surrogate (Fourier Neural Operator) serving three fluid
systems from Polymathic AI's The Well: `turbulent_radiative_layer_2D`,
`rayleigh_benard`, `shear_flow`.

Two inference tiers:

- `GET /health` — model/artifact load status
- `GET /systems` — dials, dial ranges, and metrics per served system
- `POST /predict/parametric` — fast, no-NN scalar-outcome prediction (JSON)
- `POST /predict/field` — full spatial-field prediction via the trained FNO
  (binary `.npy` transport: multipart upload in, `application/octet-stream`
  `.npy` out — see `/docs` for the exact contract)

Artifacts (checkpoint, normalization stats, parametric tables) are pulled at
startup from the public HF dataset repo `KrishMalik/deltav-fluid`.

## Public prototype

Open https://krishmalik-delta-v.hf.space for the landing page. The frontend
includes `#simulator`, `#datasets`, `#history`, `#models`, `#settings`, and
`#about` views, plus a six-step feature walkthrough and persistent light/dark themes.

The simulator separates parameter-table estimates from fixed-sample FNO
prediction. Its timeline plays the five recorded frames; it is not an
autoregressive rollout. Predictions always use t0–t3 to predict t4.
Comparison metrics are the API's full-precision relative L2, RMSE, and SSIM.
The self-imposed v1 target is strictly relative L2 < 15% for scalar and
pressure; velocities have no quantitative target. This is a release policy,
not evidence of generalization. Physical domain aspect ratios, field
orientation, numeric color scales, binary `.npy` transport, and startup
health polling are preserved.

History stores at most 20 runs in this browser's localStorage, including
metrics and small comparison previews. It does not persist full field arrays
or synchronize across devices. The landing image is a real tracer snapshot
from `samples/sf_Re1e5_Sc1.npy`, frame 4, rendered with the existing viridis
lookup table, x horizontal and y increasing upward.

### Scope of verification

On 11 September 2026, all three live parametric endpoints returned finite
outcome estimates. All six sample endpoints returned their documented
uint8 arrays and scales; all six prediction endpoints returned `(2,4,256,256)`
uint8 bundles and valid per-channel quality metrics. Measured relative L2
is now available per window and channel in `/quality` and the About table,
not a held-out benchmark. Eight of twelve primary-channel results meet the
self-imposed target; four do not. The twelve velocity scores have no target.

## Cached fixed-window predictions

Startup precomputes the six bundled windows before `/health` reports ready.
`GET /samples/{id}/predict` returns cached binary bytes, with a locked live
fallback on a cache miss. `X-Inference-Seconds` is original model computation,
not request latency. `X-Prediction-Cache` identifies a server hit/miss;
`X-Response-Seconds` measures only the endpoint handler, excluding middleware,
queueing and network transport. Warm end-to-end latency must be measured separately.

Predictions cache for one hour; `/systems`, `/samples`, and `/quality` for
five minutes; recorded sample frames for one day. `/quality` provides the
same full-precision metrics as `X-Quality` without downloading image bundles.
Use the `/health` `cache_revision` in a `?revision=` prediction URL to avoid
reusing outputs after a model/restart change. An obsolete revision returns 409.
SSIM is undefined (null) for constant reference fields; relative L2 and its
pass/fail are undefined for zero-norm references. No substitute score is invented.

Backend regression tests (from the repository root):
`python -m pytest -q tests/test_prediction_cache.py`.
Tests mock model startup and do not download checkpoints or training data.
Frontend behavior checks: `node --test tests/frontend.test.cjs`.

Iteration 2 live check (11 September 2026): startup logs confirmed all six
precomputations, cache size 6, epoch 59. A warm uncompressed curl download
took 4.923 s (first byte 0.835 s, 524,416 bytes); another compressed request
took 9.874 s, while its endpoint handler reported 0.000012 s. These are
connection-dependent observations, not a controlled benchmark. The requested
<200 ms end-to-end criterion was **not met**. Local browser reuse took 8 ms
in the preview and made no new prediction request. It must not be described
as new inference. Model computation is labeled separately in the interface.

Keep the Space's Docker configuration, samples, and model artifacts unchanged.
Never download `slices/`; runtime artifact pulls stay scoped to checkpoints
and normalization statistics.
