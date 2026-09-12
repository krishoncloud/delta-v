---
title: Delta-V
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Delta-V

**Learned physics surrogates, built one domain lane at a time.**

Live prototype: https://krishmalik-delta-v.hf.space

Delta-V is not a fluids tool. It is the scaffolding for serving, measuring, and
honestly presenting learned approximations of physical systems. Fluid dynamics
is the **first domain lane** running through that scaffolding; the tooling, the
maturity ladder, and the honesty rules are designed to carry over to the next
lanes (solid mechanics, heat transfer, electromagnetics, …) unchanged. What is
live today is lane 1.

---

## What lane 1 does

Three 2D fluid systems from Polymathic AI's [The Well](https://polymathic-ai.org/the_well/):

| System | Dials | Native grid |
|---|---|---|
| Turbulent radiative layer (`turbulent_radiative_layer_2D`) | cooling time `t_cool` | 128 × 384 |
| Rayleigh–Bénard convection (`rayleigh_benard`) | Rayleigh × Prandtl | 512 × 128 |
| Shear flow (`shear_flow`) | Reynolds × Schmidt | 256 × 512 |

Three ways to explore each one, from fastest to deepest:

1. **Parameter explorer** — turn a log-scale dial, get an instant numeric
   outcome estimate (flux and turbulent velocity) interpolated from precomputed
   tables. No neural network runs. Accuracy of this tier is *not* measured by
   the field metrics below and is an open evaluation item.
2. **Next-state prediction** — a Fourier Neural Operator takes four recorded
   frames and predicts the fifth. The prediction, the ground-truth reference,
   and the signed difference are shown side by side with per-field error.
3. **Bounded autoregressive rollout** — feed each prediction back in as input
   and generate up to 30 frames from the same four seeds. Three separate
   official-test clips provide ground truth through frame 10; the error curve
   is drawn only where a reference exists, never extrapolated.

Every prediction carries its measured error. There is no mode that hides it.

---

## Honesty rules

These apply to every lane, not just this one:

- **A number has a target or it says it has none.** The self-imposed v1 target
  is strictly relative L2 < 15 % for scalar and pressure fields. Velocity
  fields have *no* quantitative target in this release and are labelled
  "directional structure only" — not scored against a bar that was never set.
- **Undefined is undefined.** SSIM on a constant reference is `null`. Relative
  L2 on a zero-norm reference is `null`. No substitute score is invented.
- **Rollout is never boosted.** True frames are not injected mid-rollout to
  make the output look better. The whole point of the rollout view is to show
  where the model drifts; boosting would hide exactly that.
- **Static results stay static.** The Euler probe (below) reports training
  losses. It is never presented as live inference, and no Euler prediction is
  served under the joint model with a misleading label.
- **Training-set numbers and held-out numbers are kept apart** and labelled.

---

## Evidence

### Training-window check (six bundled windows)

Two windows per system, drawn from training slices. They demonstrate behaviour,
not generalization. **8 of 12** scalar/pressure scores meet the < 15 % target;
**4 do not**, and those four stay visible in the UI. The 12 velocity scores have
no target. Full per-field values appear on the About page.

### Held-out check (six official test windows)

Six fixed windows from the official `data/test/` split — outside anything the
training notebook read — evaluated through the served epoch-59 model. **All 12**
scalar/pressure scores meet the < 15 % target in this small check. No tuning
used these results. Exact source identifiers and all 24 scores:
[`static/heldout-results.json`](static/heldout-results.json).

This is a small probe, not a benchmark. Checkpoint lineage is owner-supplied,
not cryptographically established. Broad generalization is not claimed.

### Rollout drift (three official-test clips, 7 steps)

| System | Compute, 7 steps | Scalar rel. L2, f4 → f10 |
|---|---:|---:|
| Turbulent radiative layer | 13.45 s | 9.21 % → 36.86 % |
| Rayleigh–Bénard | 13.47 s | 9.05 % → 22.39 % |
| Shear flow | 13.19 s | 3.09 % → 36.63 % |

Measured on three clips on the live CPU Space; not universal rollout accuracy.
Drift need not increase monotonically and curves are never smoothed to look
like it does. Full record: [`rollout-verification.json`](rollout-verification.json).

### Euler generalization probe (static, lane-1 transfer test)

`euler_multi_quadrants_openBC` is a fourth fluid system held out entirely from
joint pretraining. Question: does prior learning on three fluids help
scarce-data fine-tuning on a fourth the model never saw?

| Run | Fine-tune epochs | Result | Status |
|---|---:|---|---|
| First | 15 | Pretrained arm ahead by 44.2 % | **Superseded — not used.** The from-scratch arm had not converged, so the gap was inflated. |
| Rerun | 30 | Pretrained final loss **0.0089** vs. from-scratch **0.0107** | **Reported.** Both arms converged. Pretrained start is **16.3 % lower error**. |

Caveats, in full:

- A file-selection bug limited fine-tuning data to **one** distinct gamma value
  instead of the four intended. It was fixed (gamma-aware selection) but the fix
  was not applied before the 30-epoch run above was measured. The planned rerun
  uses the fixed selection; the number may shift.
- **No fine-tuned checkpoint exists.** Neither arm was saved or pushed —
  confirmed directly against the HF dataset repo. Only the losses survive.
  This is why Euler is static on the site and cannot run live.
- Euler's normalization stats *do* exist on HF; no parametric table does.
- These are aggregate training losses, not the simulator's per-field relative
  L2, and not comparable to the 15 % target.

This is evidence *in the direction of* transfer to one unseen fluid system. It
is not proof of generalization to arbitrary physics.

---

## Where the bottleneck is

The served checkpoint (`domain_fluid.pt`, epoch 59) was trained at **prototype
budget on bounded slices of The Well, not on the full trajectory sets**.
Pretraining used turbulent radiative layer as the primary set with shear flow
and Rayleigh–Bénard as joint co-training sets.

That training scale — more than the FNO architecture (32 × 32 spectral modes,
128 hidden channels, 16 in / 4 out channels) — is what sets today's accuracy
and how quickly rollout error grows. The lever left on this lane is training
data and budget. The **rollout horizon** — how many frames the model can
generate from four inputs before crossing the error target — is the number
expected to move as training scales, and it is how progress on lane 1 will be
measured.

A note on the training notebook: it split sliding windows randomly rather than
by trajectory, and fit normalization statistics before the split, so its
validation loss is not trajectory-held-out. New offline tooling in
[`research/`](research/README.md) splits whole trajectories, validates
hash/exposure, and fits training-only normalization. The full re-run has not
been executed; it needs prepared data and compute.

---

## Maturity ladder (per domain lane)

**SEM → Multi-system SEM → Proto-LEM → LEM**

- **SEM** — single-system engineering model. The three live systems are
  presented at this rung. They currently share one jointly-trained checkpoint,
  but weight sharing alone does not establish the next rung.
- **Multi-system SEM** — demonstrated reliable shared learning across systems
  within the lane.
- **Proto-LEM** — demonstrated generalization to one *unseen* system in the
  lane. The Euler probe is evidence in this direction, not a claim of arrival.
- **LEM** — broad, rigorously validated model within the lane. Not claimed.

Each future domain lane repeats this climb with the same tooling and its own
measured evidence.

---

## Also shipped

- Landing page, About/limitations page, technical note, dataset and model
  pages, six-step walkthrough, light/dark themes, mobile navigation.
- Browser-local run history (up to 20 runs, no account, no server storage).
- Shareable selection links.
- Scientifically honest colormaps only: viridis (sequential) and coolwarm
  (diverging, symmetric about zero). Physical aspect ratios preserved.
- Compact display transport so the free-tier Space is usable; full-precision
  arrays are what get scored.
- Startup precomputation of the fixed windows, with a wake-up screen that
  reports loading phase while the 858 MB checkpoint loads.
- In-app feedback form. Feedback is **public**: the form prepares text in the
  browser and hands it to a GitHub issue that you publish yourself. There is no
  private inbox. Delivery was verified end-to-end ([#1](https://github.com/krishoncloud/delta-v/issues/1)).
- A [five-minute usability test kit](static/usability-test.html) for 5–10
  participants.
- Scoped security review in [`SECURITY_REVIEW.md`](SECURITY_REVIEW.md):
  bounded request bodies, input validation, single-computation admission
  control, secrets scan, dependency audit (71 packages, no advisories), XSS
  review. Not a penetration test. No per-IP rate limit exists yet.

---

## Repository layout

```
main.py               FastAPI service, startup loading, caching, serving
deltav_core.py        Canonical adapter, normalization, FNO build/load, rollout
static/               Frontend (plain HTML/CSS/JS), technical note, results JSON
samples/              Six training windows + index for three rollout clips
research/             Trajectory-separated split, evaluation, Euler three-arm
                      runner, bounded reference extractor — offline tooling
tests/                Backend (pytest) and frontend (node --test) regressions
rollout-verification.json   Live rollout check record
SECURITY_REVIEW.md    Scoped review, 2026-09-12
```

Model artifacts (checkpoint, normalization stats, parametric tables, rollout
clips) live in the public HF dataset repo `KrishMalik/deltav-fluid` and are
pulled at startup, scoped to `checkpoints/`, `stats/` and `rollout_samples/`.

---

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860. First start downloads the checkpoint; the wake-up
screen shows progress.

Tests (no downloads, model startup is mocked):

```bash
python -m pytest -q
node --test tests/frontend.test.cjs
```

## Deploy

The Space rebuilds automatically from the repository. Keep the Docker
configuration, samples and artifact scoping unchanged.

**Never download `slices/`** (92 GB of raw HDF5). Runtime pulls stay scoped to
checkpoints, stats and the three bounded rollout clips. The research tools read
source files through bounded HTTP range requests for the same reason.

**Never retrain or overwrite the production checkpoint in place.** Epoch-59
`domain_fluid.pt` is intentional; the older `_best` file was found to hold
epoch-3 weights due to a save-if-better bug and is not automatically better.

---

## Roadmap

1. **Euler, properly.** Rerun both arms with gamma-aware file selection, save
   both checkpoints with provenance, then wire a real side-by-side. Add the
   frozen-backbone learned-adapter arm (implemented, not yet run).
2. **Train at scale.** Full-trajectory, trajectory-separated training on lane
   1 and re-measure the rollout horizon. This is the main lever.
3. **Evaluate the parametric tier** on its own terms.
4. **Real user sessions** with the usability kit; feedback drives the next
   iteration.
5. **Infra follow-ups:** rate limiting, CORS allowlist, private feedback
   channel, deployment lockfile/SBOM.
6. **Open lane 2.**

---

## Credits

Built by Krish Malik and Adya Anwesha on Polymathic AI's The Well. Source:
https://github.com/krishoncloud/delta-v · Feedback (public, GitHub sign-in
required): https://github.com/krishoncloud/delta-v/issues
