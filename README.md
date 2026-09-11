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
