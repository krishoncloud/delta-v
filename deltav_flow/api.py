"""FastAPI routes for the study workflow (/flow/*) and the worker protocol (/flow/worker/*).

The API validates, authorizes, persists and serves. It never trains: fits,
predictions and recommendations are worker jobs, so a CPU-only API host never
pretends to adapt a model. Owners authenticate with a bearer token created by
POST /flow/session; workers with FLOW_WORKER_TOKEN.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import acquisition as Q
from . import adapt as A
from . import bundle as B
from . import contract as C
from . import report as R
from .store import Conflict, NotFound, Store, new_id

HERE = Path(__file__).resolve().parent
EXAMPLE_MANIFEST = json.loads((HERE / "examples" / "task8_r701.json").read_text())
EXAMPLE_DISPLAY_NAME = "Worked example: generated 2D flow study"
MAX_POINTS = 64
MAX_FIELD_POINTS = 2


@dataclass
class FlowConfig:
    data_dir: str = field(default_factory=lambda: os.environ.get("FLOW_DATA_DIR", str(HERE.parent / "work" / "flow")))
    worker_token: Optional[str] = field(default_factory=lambda: os.environ.get("FLOW_WORKER_TOKEN"))
    example_dir: str = field(default_factory=lambda: os.environ.get("FLOW_EXAMPLE_DIR", str(HERE.parent / "work" / "examples")))
    allow_test_profiles: bool = field(default_factory=lambda: os.environ.get("FLOW_ALLOW_TEST_PROFILES") == "1")
    retention_days: float = field(default_factory=lambda: float(os.environ.get("FLOW_MODEL_RETENTION_DAYS", "30")))
    adapt_enabled: bool = field(default_factory=lambda: os.environ.get("FLOW_ADAPT_ENABLED", "1") != "0")
    preadapted_dir: str = field(default_factory=lambda: os.environ.get("FLOW_PREADAPTED_DIR", str(HERE.parent / "work" / "preadapted")))
    embedded_worker: bool = field(default_factory=lambda: os.environ.get("FLOW_EMBEDDED_WORKER") == "1")
    lease_seconds: float = 120.0
    sync_wait: float = 20.0
    worker_fresh_seconds: float = 60.0


class Flow:
    def __init__(self, config: FlowConfig):
        self.config = config
        self.store = Store(config.data_dir)


_FLOW: Optional[Flow] = None
_FLOW_LOCK = threading.Lock()


def configure(config: Optional[FlowConfig] = None) -> Flow:
    global _FLOW
    with _FLOW_LOCK:
        if _FLOW is not None:
            _FLOW.store.close()
        _FLOW = Flow(config or FlowConfig())
        return _FLOW


def flow() -> Flow:
    # Startup threads (embedded worker, pre-adapted model fetch) may ask at the same time;
    # only one of them may create the store.
    if _FLOW is not None:
        return _FLOW
    with _FLOW_LOCK:
        if _FLOW is None:
            globals()["_FLOW"] = Flow(FlowConfig())
        return _FLOW


def body_limit(path: str, method: str) -> Optional[int]:
    """Per-route request body limits, enforced before parsing by main.BodyLimit."""
    if method == "POST" and (path == "/flow/studies" or (path.startswith("/flow/studies/") and path.endswith("/versions"))):
        return C.LIMITS["zip_bytes"] + 4096
    if path.startswith("/flow/worker/"):
        return 8 * 1024 * 1024
    if path.startswith("/flow/"):
        return 64 * 1024
    return None


router = APIRouter(prefix="/flow")


def rate_limit(request: Request, bucket: str):
    """Replaced by main.py with its shared limiter; a no-op when the router is used alone."""


# ---------------------------------------------------------------------- auth helpers
def bearer(request: Request) -> Optional[str]:
    header = request.headers.get("authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else None


def owner(request: Request) -> str:
    who = flow().store.owner_for(bearer(request))
    if not who:
        raise HTTPException(401, "Missing or unknown session token. Create one with POST /flow/session.",
                            headers={"WWW-Authenticate": "Bearer"})
    return who


def worker_auth(request: Request):
    token = flow().config.worker_token
    if not token:
        raise HTTPException(503, "Worker protocol disabled: FLOW_WORKER_TOKEN is not configured on the API.")
    supplied = bearer(request) or ""
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(401, "Invalid worker token")


def missing(kind):
    return HTTPException(404, f"No such {kind} for this session")


def problem(status, message, issues=None, **extra):
    return JSONResponse({"detail": message, "issues": issues or [], **extra}, status_code=status)


# ---------------------------------------------------------------------- schemas
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FitRequest(Strict):
    version_id: Optional[str] = Field(default=None, max_length=64)
    profile: str = Field(default="task8-fno-lhs-v1", max_length=40)
    allow_uncalibrated: bool = False


def _points(value):
    if not 1 <= len(value) <= MAX_POINTS:
        raise ValueError(f"Send 1 to {MAX_POINTS} query points")
    for p in value:
        if len(p) != 3 or not all(np.isfinite(p)):
            raise ValueError("Each point needs three finite controls: viscosity, forcing, initial_mix")
    return value


class PredictRequest(Strict):
    points: list[list[float]]
    fields: bool = False

    @field_validator("points")
    @classmethod
    def check_points(cls, value):
        return _points(value)


class NextRequest(Strict):
    policy: Optional[str] = Field(default=None, max_length=40)
    batch: int = Field(default=Q.DEFAULT_BATCH, ge=1, le=32)
    # Optional user limits: a [low, high] per control, inside the study's declared box.
    region: Optional[List[List[float]]] = Field(default=None, max_length=3)


class ScenarioRequest(Strict):
    basis: Literal["total_plan_including_existing", "future_runs_only"] = "total_plan_including_existing"
    planned_runs: Optional[int] = Field(default=None, ge=0, le=1_000_000)
    cost_per_run: Optional[float] = Field(default=None, ge=0, le=1e9)
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    wall_hours_per_run: Optional[float] = Field(default=None, ge=0, le=1e5)
    concurrency: Optional[int] = Field(default=None, ge=1, le=100_000)


class ChargeRequest(Strict):
    amount: float = Field(ge=0, le=1e9)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    note: str = Field(default="", max_length=200)


class WorkerClaim(Strict):
    worker_id: str = Field(max_length=120)
    kinds: list[str]
    info: dict = {}


class WorkerBeat(Strict):
    worker_id: str = Field(max_length=120)
    progress: dict = {}


class WorkerDone(Strict):
    worker_id: str = Field(max_length=120)
    result: dict = {}
    runtime_seconds: float = Field(default=0, ge=0)


class WorkerFail(Strict):
    worker_id: str = Field(max_length=120)
    error: dict
    runtime_seconds: float = Field(default=0, ge=0)
    cancelled: bool = False


# ---------------------------------------------------------------------- shared data helpers
_VERSION_CACHE: dict = {}


def load_version(version) -> B.ValidatedStudy:
    """Rebuild a stored immutable version from its content-addressed files (re-validated once per process)."""
    cached = _VERSION_CACHE.get(version["content_sha256"])
    if cached is not None:
        return cached
    s = flow().store
    base = B.validate_bundle(s.read_blob(version["files"]["base"]["sha256"]))
    ranges = [c["range"] for c in base.meta["controls"]]
    additions = [B.validate_addition(s.read_blob(a["sha256"]), ranges) for a in version["files"]["additions"]]
    study = B.combine(base, additions) if additions else base
    if study.content_sha256 != version["content_sha256"]:
        raise HTTPException(500, "Stored study files no longer match their recorded version hash")
    if len(_VERSION_CACHE) >= 4:
        _VERSION_CACHE.pop(next(iter(_VERSION_CACHE)))
    _VERSION_CACHE[version["content_sha256"]] = study
    return study


# The pinned worked-example bundle names its research archive; show a general description instead.
_ARCHIVE_NAME = re.compile(r"Delta-V Task \d+ \(protocol [^)]*\)")


def public_provenance(provenance):
    text = provenance.get("description")
    return {**provenance, "description": _ARCHIVE_NAME.sub("Delta-V research runs", text)} if isinstance(text, str) else provenance


def version_report(study: B.ValidatedStudy):
    counts = study.role_counts()
    return {"role_counts": counts, "runs": len(study.runs), "warnings": study.warnings,
            "calibrated_available": counts["calibration"] >= C.CALIBRATED_MINIMUM,
            "data_kind": study.meta["provenance"]["data_kind"], "provenance": public_provenance(study.meta["provenance"]),
            "content_sha256": study.content_sha256, "candidates": len(study.candidates.ids) if study.candidates else 0,
            "probes": len(study.probes.ids) if study.probes else 0, "ranges": [c["range"] for c in study.meta["controls"]],
            "tolerance": study.meta.get("tolerance", C.DEFAULT_TOLERANCE), "seed": study.meta.get("seed", 0),
            "acquisition": study.meta.get("acquisition"),
            "solver_seconds_supplied": sum(r.get("solver_seconds") or 0 for r in study.runs),
            "runs_with_solver_seconds": sum(1 for r in study.runs if r.get("solver_seconds") is not None)}


def latest_version(study_id):
    return flow().store.versions(study_id)[-1]


def job_view(job):
    state = job["stage"] if job["status"] == "running" and job["stage"] else job["status"]
    view = {"job_id": job["id"], "kind": job["kind"], "state": state, "status": job["status"],
            "progress": job["progress"], "error": job["error"], "attempts": job["attempts"],
            "created_at": job["created_at"], "updated_at": job["updated_at"], "finished_at": job["finished_at"],
            "study_id": job["study_id"], "version_id": job["version_id"], "model_id": job["model_id"],
            "cancel_requested": job["cancel_requested"], "runtime_seconds": job["runtime_seconds"],
            "status_url": f"/flow/jobs/{job['id']}"}
    if job["kind"] == "fit":
        view["profile"] = job["params"].get("profile")
        view["allow_uncalibrated"] = job["params"].get("allow_uncalibrated", False)
    if job["status"] == "queued":
        status = worker_status()
        online = status["available"] if job["kind"] == "fit" else status["inference_available"]
        view["queue_note"] = None if online else ("No worker for this job is connected. The job waits in the queue; "
                                                  "nothing is computed until a worker is online.")
    if job["status"] == "done" and job["kind"] in ("predict", "next"):
        view["result"] = job["result"]
    return view


def worker_status():
    workers = flow().store.workers(flow().config.worker_fresh_seconds)
    kinds = lambda w: w["info"].get("kinds", ["fit", "predict", "next", "purge"])
    return {"available": any("fit" in kinds(w) for w in workers),
            "inference_available": any("predict" in kinds(w) for w in workers),
            "workers": [{"id": w["id"], "last_seen": w["last_seen"], "device": w["info"].get("device"),
                         "profiles": w["info"].get("profiles"), "kinds": kinds(w)} for w in workers]}


def model_view(model):
    meta = model["meta"]
    m = meta["model"]
    expired = model["expires_at"] is not None and model["expires_at"] < time.time()
    # Models adapted before the labels were renamed carry the old wording; show the current one.
    label = A.PROFILES.get(model["profile"], {}).get("label", m["profile_label"])
    recipe = ({**m["recipe"], "description": C.RECIPE["description"], "profile_label": label}
              if m["recipe"].get("id") == C.RECIPE["id"] else m["recipe"])
    return {"model_id": model["id"], "study_id": model["study_id"], "version_id": model["version_id"],
            "status": "expired" if expired and model["status"] == "ready" else model["status"],
            "profile": model["profile"], "profile_label": label, "created_at": model["created_at"],
            "expires_at": model["expires_at"], "worker_id": model["worker_id"], "calibration": m["calibration"],
            "readiness": m["readiness"], "verification": m["verification"], "members": m["members"],
            "role_counts_used": m["role_counts_used"], "tolerance": m["tolerance"], "ranges": m["ranges"],
            "environment": m["environment"], "timing": m["timing"], "recipe": recipe,
            "source_sha256": m["recipe"]["source_sha256"], "version_key": m["version_key"],
            "learning": meta.get("learning", {}),
            "verification_predictions": meta.get("predictions", {}).get("verification"),
            "preadapted": meta.get("preadapted")}


def usable_model(who, model_id):
    try:
        model = flow().store.model(who, model_id)
    except NotFound:
        raise missing("model")
    view = model_view(model)
    if view["status"] == "deleted":
        raise HTTPException(410, "This model was deleted.")
    if view["status"] == "expired":
        raise HTTPException(410, "This model passed its retention period and its weights may be removed. Refit the study version.")
    return model, view


async def wait_for(who, job_id, wait):
    deadline = time.monotonic() + max(0.0, min(wait, 30.0))
    while True:
        job = flow().store.job(who, job_id)
        if job["status"] not in ("queued", "running") or time.monotonic() >= deadline:
            return job
        await asyncio.sleep(0.25)


# ---------------------------------------------------------------------- public routes
@router.post("/session", status_code=201)
def create_session(request: Request):
    rate_limit(request, "flow_session")
    owner_id, token = flow().store.create_owner()
    return {"owner_id": owner_id, "token": token,
            "note": "Keep this token private. It is the only key to your studies on this server."}


@router.get("/capabilities")
def capabilities():
    cap = C.capability()
    cap["worker"] = worker_status()
    cap["profiles"] = {k: v["label"] for k, v in A.PROFILES.items() if not v["test_only"] or flow().config.allow_test_profiles}
    cap["recommendation_policies"] = Q.POLICIES
    cap["example"] = example_status()
    cap["adapt"] = {"enabled": flow().config.adapt_enabled, "message": None if flow().config.adapt_enabled else ADAPT_OFF}
    cap["preadapted"] = preadapted_status()
    cap["limits"]["max_query_points"] = MAX_POINTS
    return cap


def example_path(kind="study"):
    entry = EXAMPLE_MANIFEST["files"][kind]
    path = Path(flow().config.example_dir) / entry["name"]
    return path, entry


def example_status():
    path, entry = example_path()
    ok = path.is_file() and path.stat().st_size == entry["bytes"]
    return {"available": ok, "example_id": EXAMPLE_MANIFEST["example_id"], "sha256": entry["sha256"],
            "build": None if ok else "Example bundle not built on this server: run scripts/pack_task8_example.py"}


def example_bytes(kind="study"):
    path, entry = example_path(kind)
    if not path.is_file():
        raise HTTPException(503, "The worked-example bundle is not built on this server. "
                                 "Run scripts/pack_task8_example.py after restoring the Task 8 archive.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise HTTPException(500, "The worked-example bundle on this server does not match its pinned SHA-256.")
    return data


@router.get("/examples/worked")
@router.get("/examples/task8")
def example_info():
    return {**EXAMPLE_MANIFEST, **example_status(), "download_url": "/flow/examples/worked/bundle",
            "label": "Generated research simulation data. Not customer CFD or OpenFOAM."}


@router.get("/examples/worked/bundle")
@router.get("/examples/task8/bundle")
def example_bundle():
    data = example_bytes()
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{EXAMPLE_MANIFEST["files"]["study"]["name"]}"',
        "X-Content-SHA256": EXAMPLE_MANIFEST["files"]["study"]["sha256"]})


ADAPT_OFF = ("Adaptation trains the model on a GPU worker, which is not connected to this public demo. "
             "The worked example includes a model adapted in advance, so you can explore it, plan next runs and read the report.")


def preadapted_status(example_id=EXAMPLE_MANIFEST["example_id"]):
    """The verified pre-adapted model installed for the worked example, if any (see scripts/import_example_model.py)."""
    root = Path(flow().config.preadapted_dir) / example_id
    try:
        info = json.loads((root / "preadapted.json").read_text())
        if not (root / "model" / "model.json").is_file() or not (root / "result.json").is_file():
            return None
        from . import preadapted as P
        if isinstance(info.get("equivalence"), dict):  # imports made before the wording changed
            info["equivalence"] = {**info["equivalence"], "basis": P.EQUIVALENCE_BASIS}
        return info
    except (OSError, ValueError):
        return None


def attach_preadapted(who, study_id, version_id, content_sha256, example_id):
    info = preadapted_status(example_id)
    if not info or info["version_key"] != content_sha256:
        return None
    result = json.loads((Path(flow().config.preadapted_dir) / example_id / "result.json").read_text())
    result["preadapted"] = {k: info[k] for k in ("example_id", "label", "device_name", "adapted_at", "manifest_sha256")}
    model_id = new_id("mdl")
    flow().store.attach_model(who, study_id, version_id, model_id, result["model"]["profile"], "preadapted", result)
    return model_id


def create_study_from(who, data, example_id=None):
    try:
        study = B.validate_bundle(data)
    except B.BundleRejected as exc:
        return problem(422, "The study bundle is not compatible with this capability.", exc.issues)
    blob = flow().store.put_blob(data)
    report = version_report(study)
    # The pinned example bundle keeps its original research name; copies opened here get a plain one.
    name = EXAMPLE_DISPLAY_NAME if example_id else study.meta["study_name"].strip()
    study_id, version_id = flow().store.create_study(who, name, report["data_kind"], blob,
                                                     study.content_sha256, report, example_id)
    if example_id:
        attach_preadapted(who, study_id, version_id, study.content_sha256, example_id)
    return JSONResponse(study_view(who, study_id), status_code=201)


@router.post("/examples/worked/start", status_code=201)
@router.post("/examples/task8/start", status_code=201)
async def example_start(request: Request):
    who = owner(request)
    data = await run_in_threadpool(example_bytes)
    return await run_in_threadpool(create_study_from, who, data, EXAMPLE_MANIFEST["example_id"])


@router.post("/studies", status_code=201)
async def upload_study(request: Request):
    who = owner(request)
    rate_limit(request, "flow_upload")
    data = await request.body()
    if not data:
        raise HTTPException(400, "Send the study ZIP as the request body (Content-Type: application/zip).")
    return await run_in_threadpool(create_study_from, who, data)


def study_view(who, study_id):
    s = flow().store
    try:
        study = s.study(who, study_id)
    except NotFound:
        raise missing("study")
    versions = s.versions(study_id)
    models = [model_view(m) for m in s.models_for_study(who, study_id)]
    jobs = [job_view(j) for j in s.jobs_for_study(who, study_id)]
    recs = s.recommendations(who, study_id)
    return {"study_id": study_id, "name": study["name"], "data_kind": study["data_kind"], "example_id": study["example_id"],
            "created_at": study["created_at"],
            "versions": [{"version_id": v["id"], "number": v["number"], "created_at": v["created_at"], "origin": v["origin"],
                          "content_sha256": v["content_sha256"], **v["report"]} for v in versions],
            "latest_version_id": versions[-1]["id"], "models": models, "jobs": jobs,
            "recommendations": [{"recommendation_id": r["id"], "version_id": r["version_id"], "model_id": r["model_id"],
                                 "created_at": r["created_at"], "rows": r["rows"], **r["meta"]} for r in recs],
            "worker": worker_status(), "capability": C.CAPABILITY_ID}


@router.get("/studies")
def list_studies(request: Request):
    who = owner(request)
    return [{"study_id": s["id"], "name": s["name"], "data_kind": s["data_kind"], "example_id": s["example_id"],
             "created_at": s["created_at"]} for s in flow().store.studies(who)]


@router.get("/studies/{study_id}")
def get_study(study_id: str, request: Request):
    return study_view(owner(request), study_id)


@router.delete("/studies/{study_id}")
def delete_study(study_id: str, request: Request):
    who = owner(request)
    try:
        models = flow().store.delete_study(who, study_id)
    except NotFound:
        raise missing("study")
    if models:
        flow().store.submit(who, "purge", {"model_ids": models}, study_id=study_id, max_active_owner=100)
    return {"deleted": study_id, "models_scheduled_for_weight_removal": models}


@router.get("/studies/{study_id}/runs")
def study_runs(study_id: str, request: Request, version_id: Optional[str] = Query(default=None, max_length=64)):
    who = owner(request)
    try:
        flow().store.study(who, study_id)
        version = flow().store.version(who, version_id) if version_id else latest_version(study_id)
    except NotFound:
        raise missing("study version")
    if version["study_id"] != study_id:
        raise missing("study version")
    study = load_version(version)
    return {"version_id": version["id"], "controls": C.CONTROL_NAMES, "outcomes": C.OUTCOME_NAMES,
            "runs": [{"run_id": r["run_id"], "role": r["role"], "controls": [r[c] for c in C.CONTROL_NAMES],
                      "outcomes": [float(v) for v in o], "solver_seconds": r.get("solver_seconds"),
                      "solver_cost": r.get("solver_cost"), "cost_currency": r.get("cost_currency")}
                     for r, o in zip(study.runs, study.outcomes)],
            "candidates": {"ids": study.candidates.ids, "controls": study.candidates.x.tolist()} if study.candidates else None}


@router.post("/studies/{study_id}/fits", status_code=202)
def start_fit(study_id: str, req: FitRequest, request: Request):
    who = owner(request)
    s = flow().store
    try:
        s.study(who, study_id)
        version = s.version(who, req.version_id) if req.version_id else latest_version(study_id)
    except NotFound:
        raise missing("study version")
    if version["study_id"] != study_id:
        raise missing("study version")
    if not flow().config.adapt_enabled:
        return problem(503, ADAPT_OFF, [{"level": "error", "code": "adapt_disabled", "message": ADAPT_OFF}])
    profile = A.PROFILES.get(req.profile)
    if not profile or (profile["test_only"] and not flow().config.allow_test_profiles):
        raise HTTPException(422, f"Unknown or disabled fit profile {req.profile!r}.")
    report = version["report"]
    if not report["calibrated_available"] and not req.allow_uncalibrated:
        return problem(422, f"{report['role_counts']['calibration']} calibration runs supplied; calibrated intervals need "
                            f"{C.CALIBRATED_MINIMUM}. Add calibration runs, or explicitly request a prediction-only fit "
                            "(allow_uncalibrated) whose spread is not calibrated.",
                       [{"level": "error", "code": "calibration_insufficient", "message": "Calibrated workflow unavailable"}])
    key = f"fit:{version['id']}:{req.profile}:{int(req.allow_uncalibrated and not report['calibrated_available'])}"
    params = {"profile": req.profile, "allow_uncalibrated": req.allow_uncalibrated and not report["calibrated_available"],
              "version": {"files": version["files"], "content_sha256": version["content_sha256"], "number": version["number"]}}
    try:
        job, created = s.submit(who, "fit", params, study_id, version["id"], new_id("mdl"), key)
    except Conflict as exc:
        raise HTTPException(409, str(exc))
    except OverflowError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "300"})
    return JSONResponse({**job_view(job), "created": created}, status_code=202 if created else 200)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    who = owner(request)
    try:
        return job_view(flow().store.job(who, job_id))
    except NotFound:
        raise missing("job")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    who = owner(request)
    try:
        return job_view(flow().store.request_cancel(who, job_id))
    except NotFound:
        raise missing("job")


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request):
    return usable_model(owner(request), model_id)[1]


async def run_model_job(request, who, model, kind, params, wait):
    s = flow().store
    try:
        job, _ = s.submit(who, kind, params, model["study_id"], model["version_id"], model["id"], max_active_owner=4)
    except (Conflict, OverflowError) as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "5"})
    job = await wait_for(who, job["id"], wait)
    view = job_view(job)
    if job["status"] == "failed":
        return problem(502 if job["error"].get("code") == "worker_error" else 409, job["error"]["message"], job=view)
    return JSONResponse(view, status_code=200 if job["status"] == "done" else 202)


@router.post("/models/{model_id}/predict")
async def predict(model_id: str, req: PredictRequest, request: Request, wait: float = Query(default=None, ge=0, le=30)):
    who = owner(request)
    model, view = usable_model(who, model_id)
    if req.fields and len(req.points) > MAX_FIELD_POINTS:
        raise HTTPException(422, f"Field responses are limited to {MAX_FIELD_POINTS} points per request.")
    lo = np.array([r[0] for r in view["ranges"]])
    hi = np.array([r[1] for r in view["ranges"]])
    x = np.array(req.points, float)
    outside = np.flatnonzero(((x < lo) | (x > hi)).any(1))
    if len(outside):
        raise HTTPException(422, f"Point {int(outside[0]) + 1} lies outside this study's declared control box "
                                 f"{view['ranges']}; extrapolation is not offered.")
    params = {"points": req.points, "fields": req.fields, "version_key": view["version_key"],
              "preadapted": (view["preadapted"] or {}).get("example_id")}
    return await run_model_job(request, who, model, "predict", params, flow().config.sync_wait if wait is None else wait)


@router.post("/models/{model_id}/next")
async def next_runs(model_id: str, req: NextRequest, request: Request, wait: float = Query(default=None, ge=0, le=30)):
    who = owner(request)
    model, view = usable_model(who, model_id)
    version = flow().store.version(who, model["version_id"])
    study = await run_in_threadpool(load_version, version)
    policy = req.policy or ("mapped-lhs-v1" if study.meta.get("acquisition") else "maximin-v1")
    if policy not in Q.POLICIES:
        raise HTTPException(422, f"Unknown policy {policy!r}; choose one of {list(Q.POLICIES)}.")
    if policy == "mapped-lhs-v1" and not study.meta.get("acquisition"):
        raise HTTPException(422, "mapped-lhs-v1 reproduces a recorded order and needs the study's acquisition settings.")
    region = None
    if req.region is not None:
        if len(req.region) != len(view["ranges"]) or any(len(r) != 2 for r in req.region):
            raise HTTPException(422, "Give one [low, high] pair per control.")
        for (lo, hi), (rlo, rhi), c in zip(req.region, view["ranges"], C.CONTROL_NAMES):
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi or lo < rlo - 1e-12 or hi > rhi + 1e-12:
                raise HTTPException(422, f"The {c} range must have low < high inside the study range {rlo} to {rhi}.")
        region = [[float(lo), float(hi)] for lo, hi in req.region]
    params = {"policy": policy, "batch": req.batch, "version_key": view["version_key"], "ranges": view["ranges"], "region": region,
              "preadapted": (view["preadapted"] or {}).get("example_id"),
              "tolerance": view["tolerance"], "acquisition": study.meta.get("acquisition"),
              "known_x": study.x().tolist(),
              "candidates": {"ids": study.candidates.ids, "x": study.candidates.x.tolist()} if study.candidates else None}
    response = await run_model_job(request, who, model, "next", params, flow().config.sync_wait if wait is None else wait)
    return response


@router.post("/studies/{study_id}/versions", status_code=201)
async def add_version(study_id: str, request: Request, parent_version_id: str = Query(max_length=64)):
    who = owner(request)
    rate_limit(request, "flow_upload")
    s = flow().store
    try:
        s.study(who, study_id)
        parent = s.version(who, parent_version_id)
    except NotFound:
        raise missing("study version")
    if parent["study_id"] != study_id:
        raise missing("study version")
    data = await request.body()

    def build():
        base = load_version(parent)
        try:
            addition = B.validate_addition(data, [c["range"] for c in base.meta["controls"]])
            combined = B.combine(base, [addition])
        except B.BundleRejected as exc:
            return problem(422, "The completed runs could not be added.", exc.issues)
        blob = s.put_blob(data)
        report = version_report(combined)
        report["added_runs"] = [r["run_id"] for r in addition.runs]
        report["added_provenance"] = public_provenance(addition.meta["provenance"])
        try:
            version_id = s.add_version(who, study_id, parent_version_id, blob, combined.content_sha256, report)
        except Conflict as exc:
            return problem(409, str(exc))
        return JSONResponse({**study_view(who, study_id), "new_version_id": version_id}, status_code=201)

    return await run_in_threadpool(build)


@router.get("/examples/worked/completions")
@router.get("/examples/task8/completions")
def example_completions(request: Request, ids: str = Query(max_length=2000)):
    """Recorded Task 8 solver outputs for requested example candidates, as an addition bundle."""
    owner(request)
    wanted = [i for i in ids.split(",") if i]
    if not 1 <= len(wanted) <= 32 or len(set(wanted)) != len(wanted):
        raise HTTPException(422, "Request 1 to 32 distinct candidate ids.")
    pool = B.validate_addition(example_bytes("recorded_pool"), C.RANGES)
    index = {r["run_id"]: i for i, r in enumerate(pool.runs)}
    unknown = [i for i in wanted if i not in index]
    if unknown:
        raise HTTPException(404, f"No recorded solver output for {unknown[:5]}. Recorded results exist only for settings our "
                                 "research runs simulated; other settings need a solver run.")
    rows = [pool.runs[index[i]] for i in wanted]
    data = B.pack("addition", {**pool.meta, "provenance": {**pool.meta["provenance"],
                  "description": "Recorded research solver outputs for the suggested worked-example runs. Not customer data."}},
                  rows, {r["run_id"]: pool.initial[index[r["run_id"]]] for r in rows},
                  {r["run_id"]: pool.result[index[r["run_id"]]] for r in rows})
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="deltav-example-completed-runs.zip"'})


# ---------------------------------------------------------------------- report, budget, artifacts
@router.put("/studies/{study_id}/scenario")
def put_scenario(study_id: str, req: ScenarioRequest, request: Request):
    who = owner(request)
    try:
        flow().store.study(who, study_id)
    except NotFound:
        raise missing("study")
    flow().store.set_scenario(who, study_id, req.model_dump())
    return get_report(study_id, request)


@router.post("/studies/{study_id}/charges", status_code=201)
def add_charge(study_id: str, req: ChargeRequest, request: Request):
    who = owner(request)
    try:
        flow().store.study(who, study_id)
    except NotFound:
        raise missing("study")
    flow().store.add_charge(who, study_id, req.amount, req.currency, req.note)
    return get_report(study_id, request)


def queried_settings(who, study_id):
    """Unique settings the owner asked the FNO about that are not supplied runs of the latest version."""
    rows = flow().store.all("SELECT params FROM jobs WHERE owner_id=? AND study_id=? AND kind='predict' AND status='done'", who, study_id)
    asked = {B.setting_key(p) for r in rows for p in json.loads(r["params"]).get("points", [])}
    if not asked:
        return 0
    supplied = {B.setting_key(r["controls"]) for r in study_runs_cached(latest_version(study_id))}
    return len(asked - supplied)


def study_runs_cached(version):
    study = load_version(version)
    return [{"controls": [r[c] for c in C.CONTROL_NAMES]} for r in study.runs]


@router.get("/studies/{study_id}/report")
def get_report(study_id: str, request: Request, format: Literal["json", "md"] = "json"):
    who = owner(request)
    view = study_view(who, study_id)
    s = flow().store
    report = R.build(view, s.ledger(who, study_id), s.scenario(who, study_id), queried_settings(who, study_id))
    if format == "md":
        return Response(R.markdown(report), media_type="text/markdown",
                        headers={"Content-Disposition": f'attachment; filename="delta-v-report-{study_id}.md"'})
    return report


@router.get("/studies/{study_id}/artifacts")
def artifacts(study_id: str, request: Request):
    who = owner(request)
    view = study_view(who, study_id)
    items = []
    for v in flow().store.versions(study_id):
        items.append({"name": f"v{v['number']}-base.zip", "kind": "study bundle", "version": v["number"], **v["files"]["base"]})
        for k, a in enumerate(v["files"]["additions"]):
            items.append({"name": f"v{v['number']}-addition-{k + 1}.zip", "kind": "completed-run addition", "version": v["number"], **a})
    items = list({i["sha256"] + i["kind"]: i for i in items}.values())
    for r in view["recommendations"]:
        items.append({"name": f"next-runs-{r['recommendation_id']}.csv", "kind": "recommended settings"})
    for m in view["models"]:
        items.append({"name": f"model-{m['model_id']}.json", "kind": "model lineage and validation",
                      "weights": [{"seed": x["seed"], "sha256": x["weights_sha256"], "bytes": x["weights_bytes"]} for x in m["members"]],
                      "weights_location": f"private worker storage ({m['worker_id']}); not downloadable from the API"})
    items += [{"name": "report.json", "kind": "report"}, {"name": "report.md", "kind": "report"}]
    return {"study_id": study_id, "artifacts": items}


@router.get("/studies/{study_id}/artifacts/{name}")
def artifact(study_id: str, name: str, request: Request):
    who = owner(request)
    view = study_view(who, study_id)
    if name in ("report.json", "report.md"):
        return get_report(study_id, request, "md" if name.endswith(".md") else "json")
    for v in flow().store.versions(study_id):
        files = [(f"v{v['number']}-base.zip", v["files"]["base"])] + [
            (f"v{v['number']}-addition-{k + 1}.zip", a) for k, a in enumerate(v["files"]["additions"])]
        for label, entry in files:
            if label == name:
                return Response(flow().store.read_blob(entry["sha256"]), media_type="application/zip",
                                headers={"Content-Disposition": f'attachment; filename="{label}"'})
    for r in view["recommendations"]:
        if name == f"next-runs-{r['recommendation_id']}.csv":
            return Response(R.recommendation_csv(r), media_type="text/csv",
                            headers={"Content-Disposition": f'attachment; filename="{name}"'})
    for m in view["models"]:
        if name == f"model-{m['model_id']}.json":
            return Response(json.dumps(m, indent=2), media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{name}"'})
    raise missing("artifact")


# ---------------------------------------------------------------------- worker protocol
def claim_impl(worker_id, kinds, info):
    s = flow().store
    s.seen_worker(worker_id, info)
    kinds = [k for k in kinds if k in ("fit", "predict", "next", "purge")]
    job = s.claim(worker_id, kinds, flow().config.lease_seconds) if kinds else None
    if not job:
        return None
    return {"id": job["id"], "kind": job["kind"], "params": job["params"], "model_id": job["model_id"],
            "study_id": job["study_id"], "version_id": job["version_id"], "attempt": job["attempts"]}


def input_impl(job_id, worker_id, sha):
    s = flow().store
    job = s.leased(job_id, worker_id)
    files = job["params"].get("version", {}).get("files", {})
    allowed = {files.get("base", {}).get("sha256"), *(a["sha256"] for a in files.get("additions", []))}
    if sha not in allowed:
        raise NotFound("Not an input of this job")
    return s.read_blob(sha)


def heartbeat_impl(job_id, worker_id, progress):
    row = flow().store.heartbeat(job_id, worker_id, progress, flow().config.lease_seconds)
    return {"cancel_requested": bool(row["cancel_requested"]), "lease_expires": row["lease_expires"]}


def complete_impl(job_id, worker_id, result, runtime):
    """Record a finished job. Raises Conflict when the worker lost its lease or returned a mismatched model."""
    s = flow().store
    job = s.leased(job_id, worker_id)
    if job["kind"] == "fit":
        meta = result.get("model", {})
        if meta.get("version_key") != job["params"]["version"]["content_sha256"]:
            s.finish(job_id, worker_id, "failed", error={"code": "version_mismatch",
                     "message": "Worker returned a model for a different study version"}, runtime_seconds=runtime)
            raise Conflict("Model/version mismatch")
        if meta.get("calibration", {}).get("status") == "uncalibrated" and not job["params"].get("allow_uncalibrated"):
            s.finish(job_id, worker_id, "failed", error={"code": "calibration_missing",
                     "message": "Calibration was required but missing"}, runtime_seconds=runtime)
            raise Conflict("Calibration missing")
        retention = flow().config.retention_days * 86400 if flow().config.retention_days > 0 else None
        s.create_model(job, worker_id, result, retention)
        s.finish(job_id, worker_id, "ready", result={"model_id": job["model_id"]}, runtime_seconds=runtime)
    elif job["kind"] == "next":
        params = job["params"]
        meta = {"policy": params["policy"], "policy_label": Q.POLICIES[params["policy"]]["label"],
                "experimental": Q.POLICIES[params["policy"]]["experimental"], "batch": params["batch"],
                "seed": (params.get("acquisition") or {}).get("seed") if params["policy"] == "mapped-lhs-v1" else Q.POOL_SEED,
                "notes": result.get("notes", []), "pool": result.get("pool"),
                "label": "Experimental recommendation. No guaranteed savings and no safe-stop assertion."}
        rec_id = s.save_recommendation(job, result.get("rows", []), meta)
        s.finish(job_id, worker_id, "done", result={**result, **meta, "recommendation_id": rec_id}, runtime_seconds=runtime)
    else:
        s.finish(job_id, worker_id, "done", result=result, runtime_seconds=runtime)


def fail_impl(job_id, worker_id, error, runtime, cancelled=False):
    flow().store.finish(job_id, worker_id, "cancelled" if cancelled else "failed",
                        error={k: error.get(k) for k in ("code", "message")}, runtime_seconds=runtime)


@router.post("/worker/claim")
def worker_claim(req: WorkerClaim, request: Request):
    worker_auth(request)
    job = claim_impl(req.worker_id, req.kinds, req.info)
    return job if job else Response(status_code=204)


@router.get("/worker/jobs/{job_id}/inputs/{sha}")
def worker_input(job_id: str, sha: str, request: Request, worker_id: str = Query(max_length=120)):
    worker_auth(request)
    try:
        return Response(input_impl(job_id, worker_id, sha), media_type="application/zip")
    except Conflict as exc:
        raise HTTPException(409, str(exc))
    except NotFound:
        raise HTTPException(404, "Not an input of this job")


@router.post("/worker/jobs/{job_id}/heartbeat")
def worker_heartbeat(job_id: str, req: WorkerBeat, request: Request):
    worker_auth(request)
    try:
        return heartbeat_impl(job_id, req.worker_id, req.progress)
    except Conflict as exc:
        raise HTTPException(409, str(exc))


@router.post("/worker/jobs/{job_id}/complete")
def worker_complete(job_id: str, req: WorkerDone, request: Request):
    worker_auth(request)
    try:
        complete_impl(job_id, req.worker_id, req.result, req.runtime_seconds)
    except Conflict as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


@router.post("/worker/jobs/{job_id}/fail")
def worker_fail(job_id: str, req: WorkerFail, request: Request):
    worker_auth(request)
    try:
        fail_impl(job_id, req.worker_id, req.error, req.runtime_seconds, req.cancelled)
    except Conflict as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True}


_EMBEDDED = {}


def ensure_preadapted():
    """Fetch and install the pre-adapted example model from a pinned Hugging Face dataset path, if configured.

    FLOW_PREADAPTED_REPO, FLOW_PREADAPTED_REVISION, FLOW_PREADAPTED_PATH and optionally
    FLOW_PREADAPTED_MANIFEST_SHA256. Failures are logged; the site then simply has no pre-adapted model.
    """
    repo = os.environ.get("FLOW_PREADAPTED_REPO")
    if not repo or preadapted_status():
        return
    from . import preadapted as P
    try:
        dest = Path(flow().config.preadapted_dir)
        dest.mkdir(parents=True, exist_ok=True)
        P.fetch_from_hub(repo, os.environ.get("FLOW_PREADAPTED_REVISION", "main"), os.environ.get("FLOW_PREADAPTED_PATH", "preadapted/task8-r701-n16-v1"),
                         dest, Path(flow().config.example_dir), os.environ.get("FLOW_PREADAPTED_MANIFEST_SHA256"))
        print("[flow] pre-adapted example model installed", flush=True)
    except Exception as exc:  # the rest of the site keeps working
        print(f"[flow] pre-adapted example model unavailable: {exc}", flush=True)


def start_background():
    """Fetch the pre-adapted model if configured, and start the in-process CPU inference worker
    when FLOW_EMBEDDED_WORKER=1 (predictions and next runs only)."""
    import threading

    from .worker import Worker

    if os.environ.get("FLOW_PREADAPTED_REPO") and not _EMBEDDED.get("fetch"):
        _EMBEDDED["fetch"] = threading.Thread(target=ensure_preadapted, name="flow-preadapted", daemon=True)
        _EMBEDDED["fetch"].start()
    if not flow().config.embedded_worker or _EMBEDDED.get("thread"):
        return
    stop = threading.Event()
    worker = Worker(LocalWorkerClient(), Path(flow().config.data_dir) / "inference-worker", device="cpu", profiles=[],
                    kinds=["predict", "next", "purge"], preadapted_root=flow().config.preadapted_dir, heartbeat_seconds=10.0)
    thread = threading.Thread(target=worker.serve, kwargs={"poll": 0.5, "stop": stop}, name="flow-inference", daemon=True)
    thread.start()
    _EMBEDDED.update(thread=thread, stop=stop, worker=worker)


def stop_background():
    if _EMBEDDED.get("stop"):
        _EMBEDDED["stop"].set()
        _EMBEDDED["thread"].join(timeout=10)
    _EMBEDDED.clear()


class LocalWorkerClient:
    """Worker protocol without HTTP, for the in-process inference worker on the API host."""

    def __init__(self, worker_id="api-cpu-inference"):
        self.worker_id = worker_id

    def _call(self, fn, *args):
        from .worker import LeaseLost
        try:
            return fn(*args)
        except Conflict as exc:
            raise LeaseLost(str(exc))

    def claim(self, kinds, info):
        return claim_impl(self.worker_id, kinds, info)

    def input(self, job_id, sha):
        return self._call(input_impl, job_id, self.worker_id, sha)

    def heartbeat(self, job_id, progress):
        return self._call(heartbeat_impl, job_id, self.worker_id, progress)

    def complete(self, job_id, result, runtime):
        return self._call(complete_impl, job_id, self.worker_id, result, runtime)

    def fail(self, job_id, error, runtime, cancelled=False):
        return self._call(fail_impl, job_id, self.worker_id, error, runtime, cancelled)
