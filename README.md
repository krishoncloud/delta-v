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
Comparison metrics are the API's full-precision relative L2 and RMSE, with
no invented acceptance threshold. Physical domain aspect ratios, field
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
across those examples was approximately 0.3–38.8%, not a held-out benchmark.

Only static frontend files need deploying for this prototype. Keep the
Space's existing backend, Docker configuration, samples, and model artifacts.
Never download `slices/`; runtime artifact pulls stay scoped to checkpoints
and normalization statistics.
