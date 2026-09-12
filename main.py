"""
Delta-V inference API.

Two tiers, per the architecture doc:
  - Fast parametric tier: sub-second scalar-outcome prediction, no NN at
    inference (interp1d / griddata over a precomputed table).
  - Deep-dive FNO tier: full-field spatial prediction via the trained model.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables (set these in your deploy target — Render/Railway/Fly):
    HF_REPO   - e.g. "KrishMalik/deltav-fluid"
    HF_TOKEN  - read-access token (repo is public, so this can be omitted
                once the repo's public visibility is confirmed permanent)
    DEVICE    - "cuda" or "cpu" (default: cpu; FNO forward pass at this
                model size is fast enough on CPU for a demo)
"""
from __future__ import annotations

import io
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from skimage.metrics import structural_similarity
from PIL import Image

import deltav_core as core

HF_REPO = os.environ.get("HF_REPO", "KrishMalik/deltav-fluid")
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional if repo is public
DEVICE = os.environ.get("DEVICE", "cpu")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", tempfile.mkdtemp(prefix="deltav_"))
CKPT_DIR = os.path.join(ARTIFACT_ROOT, "checkpoints")
STATS_DIR = os.path.join(ARTIFACT_ROOT, "stats")
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", os.path.join(HERE, "samples"))
STATIC_DIR = os.path.join(HERE, "static")

# Display metadata the frontend needs and the training code doesn't carry.
# Field units: all three Well datasets are nondimensional / code units
# (https://polymathic-ai.org/the_well/). Domains are isotropic grids so the
# grid aspect equals the physical aspect; x is horizontal, y vertical.
SYSTEM_META = {
    "turbulent_radiative_layer_2D": {
        "display_name": "Turbulent radiative layer",
        "scalar_name": "density",
        "domain": {"x": [-0.5, 0.5], "y": [-1.0, 2.0]},
        "dials": {"t_cool": {"label": "t_cool", "scale": "log", "description": "cooling time"}},
        "metrics": {
            "mass_flux": {"label": "vertical mass flux", "formula": "⟨ρ·u_y⟩"},
            "turbulent_velocity": {"label": "turbulent velocity", "formula": "rms(u′)"},
        },
    },
    "rayleigh_benard": {
        "display_name": "Rayleigh–Bénard convection",
        "scalar_name": "buoyancy",
        "domain": {"x": [0.0, 4.0], "y": [0.0, 1.0]},
        "dials": {
            "rayleigh_number": {"label": "Rayleigh", "scale": "log", "description": "Rayleigh number"},
            "prandtl": {"label": "Prandtl", "scale": "log", "description": "Prandtl number"},
        },
        "metrics": {
            "mass_flux": {"label": "vertical buoyancy flux", "formula": "⟨b·u_y⟩"},
            "turbulent_velocity": {"label": "turbulent velocity", "formula": "rms(u′)"},
        },
    },
    "shear_flow": {
        "display_name": "Shear flow",
        "scalar_name": "tracer",
        "domain": {"x": [0.0, 1.0], "y": [-1.0, 1.0]},
        "dials": {
            "reynolds": {"label": "Reynolds", "scale": "log", "description": "Reynolds number"},
            "schmidt": {"label": "Schmidt", "scale": "log", "description": "Schmidt number"},
        },
        "metrics": {
            "mass_flux": {"label": "vertical tracer flux", "formula": "⟨s·u_y⟩"},
            "turbulent_velocity": {"label": "turbulent velocity", "formula": "rms(u′)"},
        },
    },
}
UNITS_NOTE = "nondimensional (simulation code units)"

# Populated at startup — see lifespan() below.
STATE: dict = {
    "model": None,
    "stats_by": {},        # name -> {"mean":..., "std":...}
    "predictors": {},       # name -> callable with .range
    "metrics_by": {},       # name -> list of metric names
    "checkpoint_epoch": None,
    "phase": "starting",   # startup progress, surfaced by /health
    "samples": {},          # id -> metadata dict (frames loaded lazily from disk)
    "prediction_cache": {}, # id -> immutable serialized bundle + quality metadata
    "cache_revision": uuid.uuid4().hex, # invalidates browser results after any restart
}
PREDICTION_LOCK = threading.Lock()
PRIMARY_REL_L2_TARGET = 0.15  # Self-imposed v1 target; strict <, not <=.
STARTED_AT = time.time()
ANALYTICS = Counter()
ANALYTICS_LOCK = threading.Lock()
EVENTS = {"landing_viewed", "simulator_opened", "example_completed", "system_selected",
          "field_selected", "prediction_displayed", "prediction_failed", "about_opened",
          "limitations_opened", "feedback_opened"}


def _pull_artifacts():
    """Pull checkpoint + stats + parametric tables from HF into ARTIFACT_ROOT.
    Scoped to checkpoints/ and stats/ only — never widen these patterns to
    touch slices/ (92 GB of raw HDF5 training data)."""
    from huggingface_hub import snapshot_download

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    # domain_fluid.pt is loaded as primary, NOT domain_fluid_best.pt: inspecting
    # both checkpoints' saved metadata shows "_best" stopped updating after
    # epoch 3 (val_loss 0.0366) while training continued to epoch 59
    # (val_loss 0.0251, ~30% lower) — a bug in the training notebook's
    # save-if-better logic, not a naming convention to trust blindly.
    # "_best" is kept only as a fallback if domain_fluid.pt is ever missing.
    snapshot_download(
        repo_id=HF_REPO, repo_type="dataset", local_dir=ARTIFACT_ROOT,
        token=HF_TOKEN,
        allow_patterns=["checkpoints/domain_fluid.pt", "checkpoints/parametric_*.npz"],
    )
    final_path = os.path.join(CKPT_DIR, "domain_fluid.pt")
    if not os.path.exists(final_path):
        snapshot_download(
            repo_id=HF_REPO, repo_type="dataset", local_dir=ARTIFACT_ROOT,
            token=HF_TOKEN, allow_patterns=["checkpoints/domain_fluid_best.pt"],
        )
    # stats .npz files were saved under the notebook's STATS dir, not pushed
    # to HF by default in the current pipeline. If you push them (recommended:
    # add a small push step to Step 8 alongside checkpoint pushes), this will
    # pick them up. Until then, this call is a no-op if the pattern matches
    # nothing, and load_stats() will raise a clear error per-dataset below.
    try:
        snapshot_download(
            repo_id=HF_REPO, repo_type="dataset", local_dir=ARTIFACT_ROOT,
            token=HF_TOKEN, allow_patterns=["stats/**"],
        )
    except Exception:
        pass


def _load_everything():
    """Runs in a background thread so uvicorn accepts connections (and serves
    the frontend's waking-up state) while the 858 MB checkpoint downloads."""
    try:
        STATE["phase"] = "downloading artifacts"
        _pull_artifacts()

        # --- deep-dive tier: load the FNO once ---
        STATE["phase"] = "loading model"
        model = core.build_fno(device=DEVICE)
        ckpt_path = os.path.join(CKPT_DIR, "domain_fluid.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(CKPT_DIR, "domain_fluid_best.pt")
        model, epoch = core.load_fno_checkpoint(model, ckpt_path, device=DEVICE)
        STATE["checkpoint_epoch"] = epoch
        print(f"[startup] FNO loaded from epoch {epoch} on {DEVICE}")

        # --- both tiers: per-dataset stats + parametric tables ---
        STATE["phase"] = "loading tables"
        for name in core.FLUID_FAMILY:
            try:
                STATE["stats_by"][name] = core.load_stats(STATS_DIR, name)
            except FileNotFoundError as e:
                print(f"[startup] WARNING: {e}")
            try:
                table = core.load_param_table(CKPT_DIR, name)
                STATE["predictors"][name] = core.make_predictor(table)
                STATE["metrics_by"][name] = table["metrics"]
                print(f"[startup] parametric tier ready for {name}: "
                      f"{STATE['predictors'][name].range}")
            except FileNotFoundError as e:
                print(f"[startup] WARNING: {e}")

        STATE["phase"] = "precomputing predictions"
        for sample_id, meta in STATE["samples"].items():
            if meta["system"] not in STATE["stats_by"]:
                print(f"[startup] SKIP precompute {sample_id}: missing normalization stats")
                continue
            try:
                STATE["prediction_cache"][sample_id] = _build_prediction(sample_id, model)
                _sample_png(sample_id)
                print(f"[startup] precomputed prediction for {sample_id}")
            except Exception as e:
                # A failed window may be retried on demand; do not take other systems down.
                print(f"[startup] WARNING: failed to precompute {sample_id}: {e}")

        # Publish last: existing clients use model_loaded as their readiness gate.
        STATE["model"] = model
        STATE["startup_seconds"] = round(time.time() - STARTED_AT, 3)
        STATE["phase"] = "ready"
    except Exception as e:  # surfaced via /health rather than killing the server
        STATE["phase"] = f"error: {type(e).__name__}: {e}"
        print(f"[startup] FAILED: {STATE['phase']}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- sample history windows for the field tier (baked into the image) ---
    index_path = os.path.join(SAMPLES_DIR, "index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            for entry in json.load(f):
                STATE["samples"][entry["id"]] = entry
        print(f"[startup] {len(STATE['samples'])} sample windows available")
    else:
        print(f"[startup] WARNING: no samples index at {index_path}")

    threading.Thread(target=_load_everything, name="deltav-load", daemon=True).start()

    yield

    shutil.rmtree(ARTIFACT_ROOT, ignore_errors=True)


app = FastAPI(title="Delta-V Inference API", version="0.1.0", lifespan=lifespan)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict once frontend domain is known
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Scales", "X-Error-Scales", "X-Quality", "X-Inference-Seconds",
                    "X-Checkpoint-Epoch", "X-Canonical-Fields", "X-System"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ParametricRequest(BaseModel):
    system: str = Field(..., description="e.g. 'turbulent_radiative_layer_2D'")
    dials: dict[str, float] = Field(..., description="dial name -> value, e.g. {'t_cool': 0.5}")


class ParametricResponse(BaseModel):
    system: str
    dials: dict[str, float]
    metrics: dict[str, float]


class SystemInfo(BaseModel):
    system: str
    dials: list[str]
    dial_ranges: dict[str, list[float]]
    metrics: list[str]
    display_name: str
    grid: list[int]
    domain: dict[str, list[float]]
    channels: list[str]           # physical name per canonical channel
    dial_meta: dict[str, dict]
    metric_meta: dict[str, dict]
    units: str


class SampleInfo(BaseModel):
    id: str
    system: str
    dials: dict[str, float]
    t_index: int
    times: list[float]
    shape: list[int]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if STATE["model"] is not None else "loading",
        "phase": STATE["phase"],
        "device": DEVICE,
        "model_loaded": STATE["model"] is not None,
        "checkpoint_epoch": STATE["checkpoint_epoch"],
        "systems_with_stats": list(STATE["stats_by"].keys()),
        "systems_with_parametric_tier": list(STATE["predictors"].keys()),
        "prediction_cache_count": len(STATE["prediction_cache"]),
        "sample_count": len(STATE["samples"]),
        "cache_revision": STATE["cache_revision"],
        "startup_seconds": STATE.get("startup_seconds"),
    }


@app.get("/systems", response_model=list[SystemInfo])
def list_systems(response: Response):
    # Never cache a partial/empty catalog while startup is still in progress.
    if STATE["model"] is None:
        raise HTTPException(503, "model not ready", headers={"Cache-Control": "no-store"})
    response.headers["Cache-Control"] = "public, max-age=300"
    out = []
    for name, predictor in STATE["predictors"].items():
        meta = SYSTEM_META[name]
        channels = [meta["scalar_name"] if c == "scalar" else c for c in core.CANONICAL]
        out.append(SystemInfo(
            system=name,
            dials=list(predictor.range.keys()),
            dial_ranges={k: list(v) for k, v in predictor.range.items()},
            metrics=STATE["metrics_by"].get(name, []),
            display_name=meta["display_name"],
            grid=list(core.DATASETS[name]["grid"]),
            domain=meta["domain"],
            channels=channels,
            dial_meta=meta["dials"],
            metric_meta=meta["metrics"],
            units=UNITS_NOTE,
        ))
    return out


@app.get("/samples", response_model=list[SampleInfo])
def list_samples(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    return [SampleInfo(**s) for s in STATE["samples"].values()]


def _load_sample(sample_id: str) -> np.ndarray:
    if sample_id not in STATE["samples"]:
        raise HTTPException(404, f"unknown sample '{sample_id}'")
    return np.load(os.path.join(SAMPLES_DIR, f"{sample_id}.npy"), allow_pickle=False)


# ---- display transport for the frontend ---------------------------------
# Space egress is slow (~0.2 MB/s measured), so the browser gets per-channel
# uint8 quantization — the colormap resolves 256 levels, so this is visually
# lossless — in log10 space where the display is logarithmic. Quality metrics
# are computed here at full precision. The float32 API stays on /predict/field.
def _display_scales(frames_thwc: np.ndarray) -> list[dict]:
    out = []
    for c in range(frames_thwc.shape[-1]):
        v = frames_thwc[..., c]
        lo, hi = float(v.min()), float(v.max())
        canon = core.CANONICAL[c]
        if canon in ("velocity_x", "velocity_y"):
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        log = canon in ("scalar", "pressure") and lo > 0 and hi / lo > 30
        out.append({"lo": lo, "hi": hi, "log": log})
    return out


def _quantize(v: np.ndarray, sc: dict) -> np.ndarray:
    if sc["log"]:
        x = np.log10(np.maximum(v, sc["lo"])); L, U = np.log10(sc["lo"]), np.log10(sc["hi"])
    else:
        x, L, U = v, sc["lo"], sc["hi"]
    return np.clip(np.round((x - L) / ((U - L) or 1.0) * 255.0), 0, 255).astype(np.uint8)


def _npy_response(arr: np.ndarray, headers: dict) -> Response:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return Response(content=buf.getvalue(), media_type="application/octet-stream", headers=headers)


@app.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    """Display frames: uint8 .npy (N_HISTORY + 1, H, W, n_fields) — the 4 history
    frames the model sees plus the true next frame — quantized per channel with
    the scales in X-Scales (JSON: [{lo, hi, log}] per canonical channel)."""
    arr = _load_sample(sample_id)
    scales = _display_scales(arr)
    q = np.stack([_quantize(arr[..., c], scales[c]) for c in range(arr.shape[-1])], axis=-1)
    return _npy_response(q, {"X-Scales": json.dumps(scales), "Cache-Control": "public, max-age=86400"})


def _quality_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Metrics are computed before quantization, never from display pixels."""
    err = pred - truth
    quality = {}
    for c, canon in enumerate(core.CANONICAL):
        se = float((err[c] ** 2).sum())
        st = float((truth[c] ** 2).sum())
        rel_l2 = (se / st) ** 0.5 if st > 0 else None
        data_range = float(truth[c].max() - truth[c].min())
        ssim = float(structural_similarity(truth[c], pred[c], data_range=data_range)) if data_range > 0 else None
        threshold = PRIMARY_REL_L2_TARGET if canon in ("scalar", "pressure") else None
        passes = rel_l2 < threshold if threshold is not None and rel_l2 is not None else None
        note = ("Directional structure only — no quantitative target for this release."
                if threshold is None else
                "Relative L2 is undefined for a zero reference field." if rel_l2 is None else None)
        quality[canon] = {
            "rel_l2": rel_l2, "rmse": (se / err[c].size) ** 0.5,
            "ssim": ssim, "threshold": threshold, "passes": passes, "note": note,
        }
    return quality


def _build_prediction(sample_id: str, model) -> dict:
    """One shared computation path for startup precomputation and cache misses."""
    meta = STATE["samples"][sample_id]
    system = meta["system"]
    if system not in STATE["stats_by"]:
        raise HTTPException(503, "normalization statistics are not ready")
    arr = _load_sample(sample_id)
    frames = [arr[i].astype(np.float32) for i in range(core.N_HISTORY)]

    t0 = time.perf_counter()
    pred = core.run_fno_inference(model, frames, system, STATE["stats_by"][system], device=DEVICE)
    dt = time.perf_counter() - t0

    truth_c, _ = core.build_canonical(arr[core.N_HISTORY], system)
    truth = core.resample(torch.tensor(truth_c)).numpy()          # (C, 256, 256), as in training
    err = pred - truth
    quality = _quality_metrics(pred, truth)

    scales = _display_scales(arr)
    err_scales = [float(np.abs(err[c]).max()) for c in range(err.shape[0])]
    q_pred = np.stack([_quantize(pred[c], scales[c]) for c in range(pred.shape[0])])
    q_err = np.stack([_quantize(err[c], {"lo": -m, "hi": m, "log": False}) for c, m in enumerate(err_scales)])
    response = _npy_response(np.stack([q_pred, q_err]), {
        "X-Scales": json.dumps(scales),
        "X-Error-Scales": json.dumps(err_scales),
        "X-Quality": json.dumps(quality),
        "X-Inference-Seconds": f"{dt:.2f}",
        "X-Checkpoint-Epoch": str(STATE["checkpoint_epoch"]),
        "Cache-Control": "public, max-age=3600",
        "X-Cache-Revision": STATE["cache_revision"],
    })
    return {"body": response.body, "png": _png_bytes(np.stack([q_pred, q_err]).reshape(-1, 256)), "headers": {
        **dict(response.headers),
        "ETag": '"' + hashlib.sha256(response.body + json.dumps(quality).encode()).hexdigest() + '"',
    }, "quality": quality}


@app.get("/samples/{sample_id}/predict")
def predict_sample(sample_id: str, request: Request):
    """Serve a cached uint8 (2,4,256,256) bundle; compute once on a cache miss.

    X-Inference-Seconds is the original model forward-pass time, NOT request
    latency. X-Prediction-Cache distinguishes a cache hit from live computation.
    """
    t0 = time.perf_counter()
    if STATE["model"] is None:
        raise HTTPException(503, "model not ready", headers={"Cache-Control": "no-store"})
    if sample_id not in STATE["samples"]:
        raise HTTPException(404, "unknown sample")
    revision = request.query_params.get("revision")
    if revision is not None and revision != STATE["cache_revision"]:
        raise HTTPException(409, "model revision changed; reload metadata", headers={"Cache-Control": "no-store"})
    cached = STATE["prediction_cache"].get(sample_id)
    source = "HIT"
    if cached is None:
        with PREDICTION_LOCK:
            cached = STATE["prediction_cache"].get(sample_id)
            if cached is None:
                cached = _build_prediction(sample_id, STATE["model"])
                STATE["prediction_cache"][sample_id] = cached
                source = "MISS"
    headers = {**cached["headers"], "X-Prediction-Cache": source,
               "X-Response-Seconds": f"{time.perf_counter() - t0:.6f}"}
    if request.headers.get("if-none-match") == headers["ETag"]:
        headers.pop("content-length", None)
        return Response(status_code=304, headers=headers)
    return Response(content=cached["body"], media_type="application/octet-stream", headers=headers)


def _png_bytes(plane):
    buf = io.BytesIO()
    Image.fromarray(plane).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _sample_png(sample_id):
    cache = STATE.setdefault("sample_image_cache", {})
    if sample_id not in cache:
        response = get_sample(sample_id)
        array = np.load(io.BytesIO(response.body), allow_pickle=False)
        planar = array.transpose(0, 3, 1, 2)
        cache[sample_id] = (_png_bytes(planar.reshape(-1, array.shape[2])), {
            "X-Scales": response.headers["x-scales"], "X-Npy-Shape": json.dumps(list(array.shape)),
            "X-Pixel-Layout": "TCHW", "Cache-Control": "public, max-age=86400",
        })
    return cache[sample_id]


@app.get("/samples/{sample_id}/display.png")
def sample_image(sample_id: str):
    body, headers = _sample_png(sample_id)
    return Response(body, media_type="image/png", headers=headers)


@app.get("/samples/{sample_id}/prediction.png")
def prediction_image(sample_id: str, request: Request):
    response = predict_sample(sample_id, request)
    if response.status_code == 304:
        return response
    cached = STATE["prediction_cache"][sample_id]
    headers = {k: v for k, v in response.headers.items() if k not in {"content-type", "content-length", "etag"}}
    headers["X-Npy-Shape"] = "[2,4,256,256]"
    headers["X-Pixel-Layout"] = "NCHW"
    return Response(cached["png"], media_type="image/png", headers=headers)


class UsageEvent(BaseModel):
    model_config = {"extra": "forbid"}
    event: str = Field(max_length=40)


@app.post("/events", status_code=204)
def record_event(payload: UsageEvent):
    if payload.event not in EVENTS:
        raise HTTPException(422, "unsupported event")
    with ANALYTICS_LOCK:
        ANALYTICS[payload.event] += 1
    return Response(status_code=204)


@app.get("/analytics")
def usage_counts():
    # Aggregate totals only: no identifiers, IP addresses, URLs, arrays or free text.
    with ANALYTICS_LOCK:
        counts = dict(ANALYTICS)
    return {"since_unix": STARTED_AT, "counts": counts,
            "scope": "Anonymous event totals for this running server; reset on restart. Not unique people. Public counters can include automated traffic."}


@app.get("/quality")
def quality_catalog(response: Response):
    """Small evidence catalog from the same cache; no six-image download needed."""
    if STATE["model"] is None:
        raise HTTPException(503, "model not ready", headers={"Cache-Control": "no-store"})
    response.headers["Cache-Control"] = "public, max-age=300"
    return {
        "checkpoint_epoch": STATE["checkpoint_epoch"],
        "cache_revision": STATE["cache_revision"],
        "scope": "Six bundled windows from training slices; not a held-out benchmark.",
        "target_policy": "Self-imposed v1: scalar and pressure relative L2 < 0.15; velocity has no quantitative target.",
        "samples": [{
            "id": sid, "system": meta["system"],
            "quality": STATE["prediction_cache"].get(sid, {}).get("quality"),
        } for sid, meta in STATE["samples"].items()],
    }


@app.post("/predict/parametric", response_model=ParametricResponse)
def predict_parametric(req: ParametricRequest):
    """Fast tier — sub-second, no NN. REQ-4 compliant."""
    if STATE["model"] is None:
        raise HTTPException(503, f"model not ready ({STATE['phase']})")
    if req.system not in STATE["predictors"]:
        raise HTTPException(404, f"no parametric tier loaded for '{req.system}'")
    predictor = STATE["predictors"][req.system]
    try:
        result = predictor(**req.dials)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except TypeError as e:
        raise HTTPException(422, f"bad dial arguments: {e}")

    dial_out = {k: v for k, v in result.items() if k in predictor.range}
    metric_out = {k: v for k, v in result.items() if k not in predictor.range}
    return ParametricResponse(system=req.system, dials=dial_out, metrics=metric_out)


@app.post("/predict/field")
async def predict_field(
    system: str = Form(...),
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
):
    """Deep-dive tier — full spatial-field prediction via the FNO.

    Binary .npy transport instead of nested JSON: raw float32 bytes run
    ~4-5x smaller than the ASCII-decimal JSON encoding of the same array,
    which matters here since a full-resolution history can be 15-40+ MB.

    Request: multipart/form-data with `system` and EITHER `file` = a .npy
    array shaped (N_HISTORY, H, W, n_fields) float32 matching the system's
    native grid/field_map, OR `sample_id` naming one of the bundled sample
    windows (see GET /samples) — its first N_HISTORY frames are used.

    Response: application/octet-stream, a .npy array shaped (C, H, W),
    float32, denormalized, resampled to TARGET_HW — canonical field order
    and checkpoint epoch are returned as headers (X-Canonical-Fields,
    X-Checkpoint-Epoch) since the body is binary, not JSON.
    """
    if STATE["model"] is None:
        raise HTTPException(503, f"model not ready ({STATE['phase']})")
    if system not in STATE["stats_by"]:
        raise HTTPException(404, f"no normalization stats loaded for '{system}'")

    if sample_id is not None:
        meta = STATE["samples"].get(sample_id)
        if meta is None:
            raise HTTPException(404, f"unknown sample '{sample_id}'")
        if meta["system"] != system:
            raise HTTPException(422, f"sample '{sample_id}' belongs to '{meta['system']}', not '{system}'")
        arr = _load_sample(sample_id)[: core.N_HISTORY]
    elif file is not None:
        raw = await file.read()
        try:
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as e:
            raise HTTPException(422, f"invalid .npy payload: {e}")
    else:
        raise HTTPException(422, "provide either `file` (.npy upload) or `sample_id`")

    if arr.ndim != 4 or arr.shape[0] != core.N_HISTORY:
        raise HTTPException(
            422,
            f"expected shape ({core.N_HISTORY}, H, W, n_fields), got {arr.shape}",
        )

    frames = [arr[i].astype(np.float32) for i in range(arr.shape[0])]
    try:
        pred = core.run_fno_inference(
            STATE["model"], frames, system, STATE["stats_by"][system], device=DEVICE
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    buf = io.BytesIO()
    np.save(buf, pred.astype(np.float32), allow_pickle=False)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-System": system,
            "X-Checkpoint-Epoch": str(STATE["checkpoint_epoch"]),
            "X-Canonical-Fields": ",".join(core.CANONICAL),
        },
    )


# Frontend — served from the same container so there is one URL and no CORS
# hop. Mounted last so the API routes above take precedence.
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
