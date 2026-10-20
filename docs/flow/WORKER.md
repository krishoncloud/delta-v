# Adaptation worker: runtime and deployment contract

The worker (`deltav_flow/worker.py`) performs adaptation, saved-model checks,
bounded inference and recommendations. It talks to the API only through the
authenticated worker protocol, so the same code runs beside the API on one host
or on a separate GPU host. **No provider, budget or deployment has been chosen
or performed.** Those are user decisions listed at the end.

## Protocol

1. `POST /flow/worker/claim {worker_id, kinds, info}` → a job (200) or nothing (204).
   Claiming records the worker as online. A job is leased for 120 s.
2. The worker downloads each input by content hash
   (`GET /flow/worker/jobs/{id}/inputs/{sha}?worker_id=`), re-checks the hash,
   re-validates the bundle and recomputes the version hash. A mismatch fails the job.
3. A heartbeat thread posts progress every 10 s, extending the lease. The reply
   carries `cancel_requested`; a 409 means the lease was lost and the worker abandons
   the job without reporting.
4. `complete` (fit: model metadata, calibration, verification, recorded
   predictions, learning curves; predict/next: results) or `fail`
   (`{code, message}`; cancellation is reported as `cancelled`). Measured wall time
   is recorded in the study's cost ledger.

If a worker dies, its lease expires and the next claim re-runs the job
(`attempts` increments). Members with a verified `complete.json` (signature, weight
and file hashes) are reused; a partially trained member restarts from its seed.
Optimizer-step resume is not implemented. A stale worker cannot complete a job it
no longer holds (409).

## Storage

`--data-dir` holds `inputs/<sha>.zip` (cache) and `models/<model_id>/member_<seed>/`
(`weights.pt`, `contract.json`, `learning.csv`, `predictions.npz`, `result.json`,
`complete.json`) plus `model.json` and `aggregate_predictions.npz`. Adapted weights
(≈286 MB per member, ≈860 MB per model) never leave the worker; the API keeps
their hashes. Losing the disk makes predictions fail with "refit the study
version", never a fabricated answer. Purge jobs delete weights of deleted studies.
Model expiry (API `FLOW_MODEL_RETENTION_DAYS`) blocks use; deleting expired
weights is an operator task.

## Runtime

```bash
python3.11+ -m venv .venv && .venv/bin/pip install -r requirements.txt
# GPU host: install the CUDA torch wheel matching the driver instead of the CPU wheel
export FLOW_WORKER_TOKEN=...                # same secret as the API
export DELTAV_SOURCE_PATH=/path/to/domain_fluid.pt   # optional; else pinned HF download (858 MB, SHA-256 checked)
.venv/bin/python -m deltav_flow.worker --api https://<api-host> --data-dir /data/worker --device cuda
```

| Item | Measured or recorded |
|---|---|
| Recipe | 3 members × 3,000 steps, batch 2 (Task 8 `task8-fno-lhs-v1`) |
| Task 8 recorded GPU time | ≈56 s per member on the original CUDA run; round wall ≈176 s for 3 members |
| This repo's CPU measurement | Apple M3 CPU: 0.42 s/step, 0.10 s/forward, 1.8 GB RSS (≈25 min/member estimate) |
| Peak CUDA memory (Task 8) | ≈1.74 GB per member |
| Disk per model | ≈0.86 GB weights + small metadata |

## Separate GPU host checklist (pending user decisions)

- Provider, GPU type, region and monthly budget cap.
- Concurrency: the worker runs one job at a time; scale by running more workers
  against one API. The API's SQLite ledger supports one API instance.
- Private storage for API data (`FLOW_DATA_DIR`) and worker models, with backup and
  retention policy; Task 15 public/private separation before any public upload.
- Secret management for `FLOW_WORKER_TOKEN` (rotate by restarting both sides).
- TLS between worker and API; the worker only needs outbound HTTPS.
- Decide whether idle GPU hosts scale to zero; queued jobs wait safely meanwhile.

## Local development

```bash
FLOW_WORKER_TOKEN=change-me FLOW_ALLOW_TEST_PROFILES=1 .venv/bin/uvicorn main:app --port 8000
FLOW_WORKER_TOKEN=change-me .venv/bin/python -m deltav_flow.worker --api http://127.0.0.1:8000 \
    --profiles task8-fno-lhs-v1,real-source-smoke,machinery-tiny
```

`machinery-tiny` (random tiny network) and `real-source-smoke` (real source,
60 steps) exist only to exercise plumbing; every result they produce is labelled
as such and they are disabled unless `FLOW_ALLOW_TEST_PROFILES=1`.
