# Study workflow API (`/flow`)

Same-origin JSON API served by `main.py` (router in `deltav_flow/api.py`). The
API validates, authorizes, stores and serves. It never trains: fits,
predictions and recommendations are worker jobs (see [WORKER.md](WORKER.md)).

## Sessions and access

`POST /flow/session` → `{"owner_id", "token"}` (201). Send
`Authorization: Bearer <token>` on every owner route. Every study, version, job,
model, recommendation, report and artifact is scoped to its owner; other
sessions get 404. There are no accounts yet: losing the token loses access.
Rate limits: 10 sessions/min and 6 uploads/min per client.

## Routes

| Method and path | Purpose |
|---|---|
| `GET /flow/capabilities` | Supported capability, roles, recipe, source pin, limits, enabled profiles, recommendation policies, worker availability, example availability |
| `GET /flow/examples/task8` | Worked-example manifest, hashes, data label |
| `GET /flow/examples/task8/bundle` | Download the hash-verified example study ZIP |
| `POST /flow/examples/task8/start` | Create an owner-bound study from the example through the same validation/storage path as an upload. Starts no compute |
| `GET /flow/examples/task8/completions?ids=a,b` | Recorded Task 8 solver outputs for example candidates as an addition ZIP (generated research data; 404 for settings without a recorded output) |
| `POST /flow/studies` | Body: study ZIP (`application/zip`, ≤ 64 MiB). 201 study view, or 422 `{"detail", "issues": [{level, code, message, fix}]}` |
| `GET /flow/studies` | Owner's studies |
| `GET /flow/studies/{id}` | Versions (role counts, warnings, provenance, hashes), models, jobs, recommendations, worker status |
| `GET /flow/studies/{id}/runs?version_id=` | Supplied runs (controls, role, field-derived outcomes, costs) and candidates |
| `DELETE /flow/studies/{id}` | Delete study files, cancel jobs, mark models deleted, queue worker weight purge |
| `POST /flow/studies/{id}/fits` | `{"version_id"?, "profile"="task8-fno-lhs-v1", "allow_uncalibrated"=false}`. 202 new job, 200 existing identical job (idempotent), 409 another active fit, 422 calibration insufficient without opt-in, 429 queue full |
| `GET /flow/jobs/{id}` | `state` ∈ queued, loading, training, validating, calibrating, ready, done, failed, cancelled; progress (member, step, development score), error, attempts, measured runtime, queue note when no worker is online |
| `POST /flow/jobs/{id}/cancel` | Queued: cancelled now. Running: worker stops at its next check |
| `GET /flow/models/{id}` | Lineage, recipe, members (seed, weight hash, best step), calibration, readiness, verification metrics, environment, timing. 410 when deleted/expired |
| `POST /flow/models/{id}/predict?wait=` | `{"points": [[viscosity, forcing, initial_mix], …] (≤ 64), "fields": false}` (fields ≤ 2 points). Points outside the study box are rejected (422). 200 with result when the worker finishes within `wait` (default 20 s), else 202 with `status_url` |
| `POST /flow/models/{id}/next?wait=` | `{"policy"?, "batch"=8}`. Default policy `mapped-lhs-v1` when the study records it (worked example), else `maximin-v1` |
| `POST /flow/studies/{id}/versions?parent_version_id=` | Body: addition ZIP. 201 with `new_version_id`; 409 if the parent is not the latest version; 422 duplicates/incompatible |
| `PUT /flow/studies/{id}/scenario` | Budget scenario inputs `{basis, planned_runs, cost_per_run, currency, wall_hours_per_run, concurrency}`; returns the report |
| `POST /flow/studies/{id}/charges` | Record a known incremental charge `{amount, currency, note}` |
| `GET /flow/studies/{id}/report?format=json\|md` | Measured usage, model lineage, verification, recommendations, scenario, research evidence, limitations |
| `GET /flow/studies/{id}/artifacts` and `/artifacts/{name}` | Manifest and downloads: version bundles/additions, next-run CSVs, model lineage JSON, report. Weights stay on private worker storage |

Worker routes (`Authorization: Bearer $FLOW_WORKER_TOKEN`):
`POST /flow/worker/claim`, `GET /flow/worker/jobs/{id}/inputs/{sha}?worker_id=`,
`POST /flow/worker/jobs/{id}/heartbeat|complete|fail`.

## Result semantics

- Prediction results: `mean`, `sd` (raw ensemble SD, ddof 1), `raw_lower/upper`
  (normal 90% from raw SD, labelled uncalibrated), and `lower/upper` only when the
  model is calibrated (normalized joint split conformal at 0.9, marginal over
  iid queries). `members` holds per-member outputs.
- Recommendations: `rows[]` with `candidate_id`, `controls`, `reason`,
  `gap_fraction_of_diagonal`, `predicted`, `raw_sd`, optional
  `calibrated_interval`; plus `policy`, `seed`, `batch`, `notes` (exhaustion,
  non-default batch, experimental status) and a no-guarantee label.
- Reports separate measured usage, the customer scenario and historical
  research evidence. "Verified cost savings: not yet established for this study"
  is always present.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FLOW_DATA_DIR` | `work/flow` | SQLite ledger and content-addressed bundles. Mount a private persistent volume |
| `FLOW_WORKER_TOKEN` | unset (worker routes 503) | Shared secret with the worker |
| `FLOW_EXAMPLE_DIR` | `work/examples` | Built example bundles |
| `FLOW_ALLOW_TEST_PROFILES` | unset | `1` enables `machinery-tiny` and `real-source-smoke` (never in production) |
| `FLOW_MODEL_RETENTION_DAYS` | 30 | Model expiry; 0 disables |
