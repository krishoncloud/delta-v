# Delta-V — handoff to Codex

Paste everything below the line into Codex as the task brief. It's self-contained.

---

## Context

Delta-V is a physics-simulation surrogate (Fourier Neural Operator) for three
fluid systems from Polymathic AI's *The Well*: `turbulent_radiative_layer_2D`,
`rayleigh_benard`, `shear_flow`. Training is done. A working FastAPI backend
+ frontend is deployed and live:

- **Live app:** https://krishmalik-delta-v.hf.space
- **HF Space (source of truth for deploy):** https://huggingface.co/spaces/KrishMalik/delta-v (Docker SDK, HF PRO required for `cpu-basic` hardware — already subscribed)
- **HF dataset repo (model artifacts):** https://huggingface.co/datasets/KrishMalik/deltav-fluid (public)
- **Code repo:** https://github.com/krishoncloud/delta-v (public)

### To deploy changes yourself

```bash
pip install -U huggingface_hub
hf auth login          # paste a token from https://huggingface.co/settings/tokens (write scope)
```

Then push any changed files straight to the Space (it rebuilds automatically):

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    repo_id="KrishMalik/delta-v", repo_type="space", folder_path=".",
    allow_patterns=["main.py", "deltav_core.py", "static/*", "samples/*",
                     "requirements.txt", "Dockerfile", "README.md"],
    commit_message="describe the change",
)
```

Poll `https://huggingface.co/spaces/KrishMalik/delta-v` or `GET /health` on
the live URL to confirm the rebuild landed.

## Task 0 — resolve the accuracy TBD before touching the frontend

This ran through a 5-advisor/peer-review council pass before this handoff was
finalized. Verdict, unanimous across every reviewer: **do this before the
wireframe redesign, not after** — it's roughly a half-day of work, not a
week's detour, and it's the one open item most likely to actually get
questioned by an evaluator.

Why it matters: the project's SRS leaves the accuracy bar as an explicit,
undecided TBD (see "Known limitations" below). Right now the frontend shows
real per-channel error (3–39% relative L2 depending on channel/system) next
to ground truth with no stated target for what counts as acceptable. A
*more polished* site sitting on top of that undefended number is worse, not
better — it reads as confident packaging around an unaddressed gap, and
invites exactly the follow-up question a landing page can't answer ("why is
that acceptable?").

**Do this first:**
1. Replace "TBD" with a concrete, even if self-imposed, threshold per
   channel — e.g. `<15% relative L2 on pressure/density (primary) channels;
   velocity channels explicitly flagged as directional-only, not
   production-grade`. Base the actual numbers on what `/samples/{id}/predict`
   already reports (`X-Quality`), not on guesses.
2. Wire that threshold into the existing quality display (the
   `renderQuality()` panel in `static/app.js`, and/or `main.py`'s
   `/samples/{id}/predict` `X-Quality` payload) so it visibly reports
   pass/fail per channel against the stated bar, not just a bare percentage.
3. Only then move to Task 1 (wireframe redesign) and Task 2 (landing/about
   pages) below — and when you do, explicitly brief yourself (or whoever
   continues this) to **preserve the honest error/threshold display**, not
   smooth it over with confident marketing copy. A landing/about page that
   undersells the real accuracy gap is the specific failure mode the council
   flagged as worse than not having the pages at all.

Do not use this task as an excuse to expand scope (e.g. a multi-system
spike, retraining, new datasets) — that was explicitly considered and
rejected as the wrong move for this week. Keep it to writing the threshold
and wiring the display.

## Guardrail — do not violate

The dataset repo has a `slices/` folder (**92 GB** of raw HDF5 training data).
**Never** widen any `allow_patterns` / `snapshot_download` call to touch
`slices/` — every artifact pull in `main.py` must stay scoped to
`checkpoints/`, `stats/`, and the pre-extracted `samples/` already baked into
the Space image. If you need more sample initial conditions, see "Known
limitations" below for how the existing ones were extracted.

## A real bug we found and fixed — know this before touching checkpoints

`checkpoints/domain_fluid_best.pt` was **not actually the best checkpoint**.
Its "save if validation improves" logic stopped firing after epoch 3
(val_loss 0.0366) even though training continued to epoch 59 with val_loss
0.0251 (~30% lower), saved separately as `domain_fluid.pt`. We verified this
by inspecting each checkpoint's saved `{epoch, history, val_history}` dict
directly, then re-ran inference with both and confirmed epoch 59 predicts
measurably better on every channel but one (near-flat there).

**This has already been fixed at the source**: `checkpoints/domain_fluid_best.pt`
on the HF dataset repo has been overwritten with the actual epoch-59 weights,
so the filename is trustworthy again. `main.py` loads `domain_fluid.pt` as
primary with `domain_fluid_best.pt` as a fallback — both now point at the
same, correct weights. No other checkpoint files exist in the repo (checked).
If you retrain, keep this checkpoint-saving logic in mind — whatever bug
caused `_best` to stop updating in the original training notebook is still
presumably present there.

## Known limitations (say these, don't silently paper over them)

- **Model accuracy is real and uneven.** On the epoch-59 checkpoint,
  relative-L2 error against ground truth ranges from ~3% (pressure) to ~39%
  (velocity_x on the turbulent radiative layer). This is the model, not a
  bug — show it honestly (the current field-simulation view already does:
  ground truth | prediction | error, with per-channel rel-L2/RMSE computed
  server-side).
- **Only 6 sample initial conditions exist** (2 per system), baked into the
  Space image under `samples/*.npy` — each is 5 consecutive real frames (4
  history + 1 ground-truth next frame) pulled from specific `.hdf5` files in
  `slices/` via HTTP range reads (`huggingface_hub.HfFileSystem` + `h5py`,
  reading only the needed trajectory/timestep window, never the whole file).
  That extraction script wasn't saved — reconstruct it if you need more
  samples; it's straightforward (see `main.py`'s `SYSTEM_META` for the
  per-system field ordering: `[scalar, pressure, velocity_x, velocity_y]`
  mapped from each dataset's native fields — density/buoyancy/tracer plus
  pressure and velocity).
- **No autoregressive rollout.** The field tier predicts one step (t4 from
  t0–t3) and stops. Feeding the prediction back in for t5, t6... would make
  a real "simulation" animation, but this checkpoint will visibly drift
  after a few steps — worth adding as an honest demo, not a polished one.
- **Cold start**: HF Spaces free `cpu-basic` sleeps after ~48h idle. First
  request after sleep takes 1–2 min (858 MB checkpoint download + load) —
  `main.py` loads the model in a background thread and exposes progress via
  `GET /health` (`phase`: `downloading artifacts` → `loading model` →
  `loading tables` → `ready`); the current frontend's wake-up screen polls
  this. Keep that contract if you touch startup.
- **Units**: all three Well datasets are nondimensional / code units (see
  https://polymathic-ai.org/the_well/ per-dataset pages) — never invent SI
  units for `mass_flux` / `turbulent_velocity` or the fields.
- **Field-tier payload**: the API is binary (`.npy` multipart upload +
  `.npy` octet-stream response, not JSON) because raw float32 bytes run
  ~4-5x smaller than JSON's ASCII-decimal encoding, and the sample/display
  endpoints (`GET /samples/{id}`, `GET /samples/{id}/predict`) additionally
  quantize to uint8 per-channel + gzip, because Space egress bandwidth is
  slow (~0.2 MB/s measured) — keep this transport, don't revert to JSON.

## API surface (unchanged, build the new frontend against this)

- `GET /health` → `{status, phase, device, model_loaded, checkpoint_epoch, systems_with_stats, systems_with_parametric_tier}`
- `GET /systems` → per system: `dials`, `dial_ranges`, `metrics`, `display_name`, `grid`, `domain` (physical x/y extent), `channels` (physical field names), `dial_meta` (label + log/linear scale hint), `metric_meta` (label + formula), `units`
- `GET /samples` → list of bundled initial conditions: `{id, system, dials, t_index, times, shape}`
- `GET /samples/{id}` → uint8-quantized `.npy` (5, H, W, 4) native-resolution history+truth, gzip'd; scales in `X-Scales` header (JSON list of `{lo, hi, log}` per channel)
- `GET /samples/{id}/predict` → runs the FNO, returns uint8 `.npy` (2, 4, 256, 256) = `[prediction, error]`; `X-Error-Scales`, `X-Quality` (per-channel `{rel_l2, rmse}`), `X-Inference-Seconds`, `X-Checkpoint-Epoch` headers
- `POST /predict/parametric` → JSON `{system, dials}` → `{system, dials, metrics}` (instant, no NN)
- `POST /predict/field` → multipart `{system, file|sample_id}` → binary `.npy` (4, 256, 256) prediction, for programmatic/API use outside the bundled samples

## Task 1 — redesign the frontend to this wireframe, with dark mode

*(the wireframe image is attached to this brief — describe it inline for
Codex if it can't see images: left sidebar nav — Simulator / Run History /
Datasets / Models / Settings / About; top bar with 4 dropdowns — Domain
Lane / Parameters Available / Trajectory / Time Step — plus a "Run
Prediction" button; a "Current State (Ground Truth)" panel with a large
field render, colorbar, zoom/fullscreen/download controls, and an info
card; a "Prediction Comparison" panel below it with 3 side-by-side frames —
Predicted / Ground Truth / Difference — and a metrics row (MAE, RMSE, Max
Error, SSIM, Correlation); a right-hand "Run History" rail showing past runs
as thumbnail cards; a bottom timeline scrubber with play/pause and frame
rate control.)*

Rebuild the current frontend (`static/index.html`, `app.css`, `app.js`) to
match this structure and layout, adapted to what the live API actually
supports:

- **Keep** the current dark palette as the dark-mode default (cosmic indigo
  `#35235C` / ion violet `#9B7BFF` / plasma mint `#73F5C2` on near-black
  `#0B0A12`, Geist + Geist Mono) — it was chosen deliberately, don't
  reinvent it. **Add a light mode** (the wireframe shows a sun/moon toggle
  in the top bar) — same structure, light surface tokens.
- Left nav matches the wireframe's items where they map to something real:
  **Simulator** (the current dial + field view), **Datasets** (the 6 bundled
  sample initial conditions plus the 3 systems' dial ranges/metrics — read
  from `/systems` and `/samples`), **About** (see Task 2). "Run History" and
  "Models" imply state the backend doesn't have (no run persistence, no
  multi-model registry) — either stub them honestly as "coming soon" or
  scope them to session-local history (localStorage) rather than pretending
  there's a backend for them. Don't fabricate data to fill the UI.
- Wireframe's `SSIM` / `Correlation` metrics aren't computed by the API
  today — either add them server-side in `main.py`'s
  `/samples/{id}/predict` (alongside the existing `rel_l2`/`rmse` in
  `X-Quality`) or drop them from the UI; don't hardcode fake numbers.
- Keep the existing physical correctness: log-scale dials for
  `rayleigh_number`/`reynolds` (span 4-5 decades), correct domain aspect
  ratio per system (shear_flow is 1:2, not square), viridis for scalars
  (log₁₀ where the field spans >1.5 decades) / coolwarm symmetric-about-zero
  for velocity, real min/max on every colorbar. These came from checking
  https://polymathic-ai.org/the_well/ dataset pages directly — don't
  regress them for the sake of matching the wireframe's generic "High/Low"
  colorbar labels.
- Keep the binary `.npy` + quantized-uint8 transport (see "Field-tier
  payload" above) and the wake-up/cold-start handling (`GET /health`
  polling) — both are load-bearing for a free-tier HF Space.

## Task 2 — landing page + explanation ("About") page

Two things currently missing entirely. The project has a formal SRS
(`Delta-V_SRS_IEEE.docx`, not in this repo — ask the user for it if you need
the full document) — ground this page in it, not in ad-hoc framing:

1. **Landing page** — what a visitor sees before diving into the simulator.
   One clear statement of what Delta-V is, a way in to the Simulator, and
   enough visual interest (a live or pre-rendered field render) to signal
   this is a real physics tool, not a form.
2. **Explanation / About page**, covering, honestly:
   - **What Delta-V is** — per the SRS: "a software platform that serves
     and visualizes learned surrogate models of physical systems." An
     FNO-based surrogate for 2D fluid PDEs, trained on Polymathic AI's
     *The Well*; two inference tiers (instant parametric interpolation vs.
     full-field neural prediction).
   - **Scope, stated explicitly in the SRS — carry these framings over
     verbatim in spirit**: 2D only (not 3D) for this release; The Well is
     the only data source; **"the surrogate complements such solvers rather
     than replacing them"** — do not oversell it as CFD replacement;
     positioned as a **single-tenant internal/demonstration system**, not a
     public multi-user product — no accounts, no auth, nothing here should
     imply otherwise; outputs must always be **labelled as learned
     approximations shown alongside ground truth**, never presented as
     validated simulation.
   - **Maturity framing**: the SRS places Delta-V at "SEM" (single-system
     engineering model) stage — one system per model today — with
     multi-system SEM → proto-LEM → LEM (still fluid-dynamics only, no
     cross-domain claim) as explicit *future* stages, not current
     capability. Say this plainly rather than implying more generality
     than exists.
   - **On accuracy — do not invent a pass/fail bar.** The SRS's own TBD
     appendix leaves "quantitative prediction-quality target (e.g. relative
     error threshold)" **explicitly undecided** (TBD-3), same for the
     parametric-vs-field accuracy relationship (TBD-7). So report the real
     measured numbers (3–39% relative L2 depending on channel/system, live
     from `/samples/{id}/predict`'s `X-Quality`) as *current measured
     performance*, not as a pass or fail against some target — because
     there isn't one yet. That's accurate to the project's own spec, not a
     limitation to apologize for.
   - **What's been tried and why** — FNO chosen for resolution invariance
     and spectral efficiency on PDE-like fields; three systems from *The
     Well* for diversity of dynamics (radiative cooling, buoyancy-driven
     convection, shear-driven mixing); `euler_multi_quadrants_openBC` was
     trained on as a held-out generalization probe but deliberately isn't
     served (not part of `FLUID_FAMILY`) — a research check, not a served
     system.
   - **Dataset + time-instance explanation** — physical domain, grid
     resolution, timestep spacing per system (from `GET /systems`'
     `domain`/`grid`, and `deltav_core.DATASETS`). State clearly:
     `N_HISTORY=4` frames in, 1 frame out, all resampled to 256×256 for the
     FNO regardless of native grid.
   - **Non-functional targets that ARE fixed in the SRS** — worth stating
     as promises being kept: parametric tier sub-second response (NFR-1);
     field tier fast enough for interactive use, no GPU required for a
     single request (NFR-2/3). Both true today — parametric is instant,
     field inference is ~0.4–1.8s on the Space's CPU.
   - **A feature walkthrough** — screenshots or an inline guided tour of
     the actual Simulator UI (dial controls → Predict → field simulation →
     reading the comparison panel), not a generic feature list.
   - **Error messaging** (SRS REQ-5/13/26): invalid parameters, unavailable
     data, or a failed prediction must surface as clear, non-technical
     messages — audit the current frontend's error states against this;
     today's `predict()`/`runField()` mostly just show the raw API error
     string, which may not qualify as "non-technical."

Ground every number and claim in the live API, the verified facts above, or
the SRS — nothing invented, nothing carried over from earlier (wrong)
assumptions.

## Backlog items visible in the SRS but not yet built

Worth knowing these are *specified*, even though out of scope for this
handoff unless you have time: a **run history view** listing past prediction
runs for retrieval (REQ-33–35) — the wireframe's "Run History" rail assumes
this; today there's no persistence layer for it, so either scope it to
`localStorage` (session-local, honest) or treat it as a stub. A
**parameter-metadata endpoint** exposing valid ranges/discrete values and
units per dial (REQ-1) — `GET /systems`' `dial_ranges`/`dial_meta` already
covers most of this. **Time-step navigation** through a trajectory (REQ-31)
— the current timeline scrubber covers t0–t4 of one sample; a longer
trajectory isn't available without pulling more frames from `slices/`
(same extraction approach as the existing samples, same guardrail applies).

## Design references already in the repo

- `design/delta-v-dashboard.html` — an earlier mockup, now superseded by
  the live app; useful for the color system, not layout (the wireframe
  above takes priority on structure).
- `static/app.css` — the current token set (`--bg`, `--surface`, `--line`,
  `--text`, `--accent`, `--live`) to extend for light mode rather than
  replace.
