# Tasks 10/11 validation record

Branch `task10-11-fno-workflow` (local only), based on
`Delta-V-1.1-20th-October-2026` at `a734a62`. Environment: macOS, Apple M3,
8 GB RAM, Python 3.13.0 in `.venv`, torch 2.11.0 (CPU), neuraloperator 2.0.0,
numpy 2.1.2, scipy 1.14.1, FastAPI 0.141.1, Pydantic 2.13.5, Node 22.13.1.

Each check below states what kind of evidence it is. Do not read a lower tier as
a higher one.

| Tier | Meaning |
|---|---|
| Unit/mocked | Pure logic or fake HTTP; no model |
| Machinery (tiny) | `machinery-tiny` profile: randomly initialised tiny FNO. Proves plumbing only |
| Full-size source smoke | Real pinned 858 MB source and real architecture, a few optimizer steps |
| Full recipe, local CPU | Real source, real `task8-fno-lhs-v1` recipe, on this M3 CPU |
| Full recipe, GPU | **Not run.** No GPU worker selected |
| Hosted deployment | **Not done.** No deployment, purchase or public push |

## Automated tests

| Command | Result | Tier |
|---|---|---|
| `node --test tests/*.test.cjs` | 25 passed | unit/mocked (front end) |
| `.venv/bin/python -m pytest -q tests/test_flow_bundle.py` | 28 passed | unit (validator, real example bundle) |
| `.venv/bin/python -m pytest -q tests/test_flow_engine.py` | 12 passed, 1 skipped | machinery; includes bit-identical equivalence with the notebook's `fit_member7` |
| `DELTAV_SOURCE_PATH=<verified blob> pytest -k full_size tests/test_flow_engine.py` | 1 passed (16.6 s, 1.78 GB RSS) | full-size source smoke |
| `.venv/bin/python -m pytest -q tests/test_flow_api.py` | 15 passed | machinery through API + worker protocol |
| `.venv/bin/python -m pytest -q tests/test_flow_evidence.py` | 7 passed | unit (evidence numbers vs published CSVs, budget formulas, acquisition) |
| Existing API, research and notebook suites | see final run below | regression |

## What the tests establish

- **Input/schema**: valid example and custom bundles; malformed ZIP, bad NPY,
  traversal/absolute/backslash/dot paths, symlinks, duplicate entries,
  decompression ratio, oversized members, archive/unpacked/file-count limits,
  413 for oversized bodies, pickle/object arrays, wrong dtype/shape/order,
  NaN; incompatible grid/BC/time/units/geometry/channels/layout; out-of-range and
  nonfinite controls; missing/unreferenced/shared fields; duplicate IDs and
  settings within and across roles; insufficient role counts with required
  changes; scalar outcome mismatches; non-analytic initial fields; checksum
  tampering; candidates carrying outcomes; unexpected files.
- **Scientific path**: recipe constants equal the Task 8 locked protocol;
  extracted `fit_member` reproduces `fit_member7` exactly (weights, predictions,
  best step, dev score, normalizers); train-only normalizers; member predictions
  identical when calibration/verification labels change; source state drives the
  fit; reload and batch-partition invariance; time-capped member cannot seal a
  model; uncalibrated output requires opt-in and has no conformal rule; source
  digest checked and unchanged.
- **Jobs/storage/access**: idempotent fit submission; one active fit per owner;
  restart of the API on the same data directory; worker crash (expired lease)
  re-claimed with attempts = 2, stale worker rejected; running-job cancellation;
  verified members reused and a tampered member refit; ownership 404s on every
  owner route; worker and owner tokens not interchangeable; version mismatch and
  tampered weights fail with actionable messages; deletion purges worker weights.
- **Recommendations**: exact recorded mapped-LHS order for repetitions 701, 1409
  and 2111; `maximin-v1` deterministic, unique, feasible, label-free; batch size
  honoured or exhaustion explained; recorded next runs for the example through
  the API equal `r701-00078, 00277, 00437, 00276, 00059, 00239, 00289, 00070`;
  added runs create a new version that the next fit uses.
- **Budget/marketing**: E/B/A formulas (3,280 and −720 brief examples); missing
  cost → no currency amount; unknown charges not counted as zero; mixed
  currencies flagged; every displayed evidence number traced to the CSVs; no
  forbidden claims in the page copy; live scripts contain no browser model,
  synthetic solver or mock fixture.

## Browser verification (in-app browser, real API + worker on localhost)

Machinery profile, worked example: example opened with no compute started; fit
queued with "no worker connected" note; page refresh restored study and job from
the server; worker completed the fit; model panel with 200-run verification and
parity chart; Explore curve (21 worker predictions), point query with predicted
fields, out-of-box input refused; Next runs equal the recorded order; recorded
completions created version 2 (24 train runs); report with measured usage,
scenario, research evidence; invalid upload shows server issue and fix; 375 px
mobile width without horizontal scrolling on home, evidence and workbench.

## Full recipe on the worked example (GPU, MoLab)

Run by the user with `research/task10_11/DeltaV_ExampleAdapt_MoLab.py` on an
NVIDIA RTX PRO 6000 Blackwell Server Edition (torch 2.11.0, numpy 2.4.6, scipy 1.18.0):

- 3 members × 3,000 steps, 159 s in total; best steps 1600, 2300 and 1000.
- All three weight files are bit-identical to the weights recorded by the original
  Task 8 run (r701, fno_lhs, 16 runs), and the stored predictions on all 879 query
  settings equal the recorded ones exactly.
- Study-specific verification on the 200 held-out runs: 200/200 within ±0.015,
  RMSE 0.003384 (KE) and 0.003245 (pressure RMS), joint coverage 0.915, the same
  calibration rule (q = 2.262, rank 36 of 39) as recorded.
- The model's recorded study key (`39896192…`) differs from this site's example key
  (`444b8dfa…`); the cause was not reproduced (the same bundle gives `444b8dfa…`
  under both numpy 2.1.2 and 2.4.6). The importer accepts it only through
  `equivalent_to_recorded`: recorded weights, identical training runs, a CPU spot
  check (max difference 3.5e-5) and verification metrics recomputed from the
  site's copy. Both keys are kept in `preadapted.json` / `model.original.json`.

Local CPU serving of this model (Apple M3): 21-point curve 3–3.6 s with chunked
batching (outputs within 9e-8 of per-row prediction), one setting about 1 s,
next runs about 3 s.

An earlier local-CPU run of the same recipe was cancelled after member 1, whose
development score tracked the recorded run (0.000604 vs 0.000611 at step 1600).

## Not verified

- Hosted deployment; a live GPU worker for the public site (adaptation is off there by design).
- Python 3.11 (Docker base) and the Docker image build: not run here.
- Task 9 real solver/customer/OpenFOAM data; Task 15 privacy separation.
- Multi-instance API (SQLite ledger is single-instance by design).
- Prospective run or cost savings of any recommendation policy.
