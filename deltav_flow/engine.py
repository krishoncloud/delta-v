"""Conditioned FNO adaptation engine extracted from research/DeltaV_Task8_MoLab.py.

Scientific semantics are unchanged from `fit_member7`/`predict_member7` and the
Task 7 conformal helpers: same model, encoding, loss, optimizer groups, cosine
schedule, minibatch RNG, development checkpoint selection, per-row prediction,
reload and batch-invariance checks. Differences are plumbing only: inputs are
arrays from a validated study instead of a pandas view, progress/cancellation
callbacks exist, and query roles are explicit.

No HTTP code lives here. The worker calls these functions outside request handling.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import contract as C
from .physics import initial_fields, physical_outcomes

SOURCE_CHUNK = 8 << 20


class Cancelled(RuntimeError):
    pass


class MemberIncomplete(RuntimeError):
    """A member hit its safety cap or failed a check; it must never become part of a ready model."""


# --------------------------------------------------------------------------- files
def digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(SOURCE_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# --------------------------------------------------------------------------- source
def resolve_source(local_path: str | None = None, source=C.SOURCE) -> Path:
    """Verified path of the pinned source checkpoint. Downloads only that file at that revision."""
    if local_path:
        path = Path(local_path).expanduser()
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=source["repo"], filename=source["filename"], revision=source["revision"],
                                    repo_type="dataset", token=False))
    if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
        raise ValueError("Source checkpoint size/SHA-256 mismatch; refusing to load.")
    return path


def load_source(path, source=C.SOURCE):
    """Weights-only load of the verified source backbone state. Never writes to the source file."""
    from neuralop.layers.spectral_convolution import SpectralConv

    path = Path(path)
    if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
        raise ValueError("Source checkpoint size/SHA-256 mismatch; refusing to load.")
    with torch.serialization.safe_globals([torch.nn.functional.gelu, SpectralConv]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = {k: v.clone() for k, v in checkpoint["model"].items() if isinstance(v, torch.Tensor)}
    lineage = {"source": dict(source), "epoch": int(checkpoint["epoch"]), "loaded_tensor_keys": len(state)}
    del checkpoint
    return state, lineage


# --------------------------------------------------------------------------- model
def normalize_training(x, initial, fields, y, ids):
    norm = {"train_ids": list(ids), "x_mean": x.mean(0).tolist(), "x_std": np.maximum(x.std(0), 1e-06).tolist(),
            "outcome_scale": np.maximum(y.std(0), 1e-06).tolist()}
    for name, arr in [("initial", initial), ("target", fields)]:
        norm[name + "_mean"] = arr.mean((0, 2, 3)).tolist()
        norm[name + "_std"] = np.maximum(arr.std((0, 2, 3)), 0.0001).tolist()
    return norm


class TransferFNO(torch.nn.Module):
    def __init__(self, nparams, architecture, coordinates="periodic"):
        from neuralop.models import FNO

        super().__init__()
        self.coordinates = coordinates
        self.conditioner = torch.nn.Conv2d(4 + nparams + (4 if coordinates == "periodic" else 2), 16, 1)
        self.backbone = FNO(**architecture)

    def forward(self, x):
        return self.backbone(self.conditioner(x))


def encode(initial, x, norm, device, coordinates="periodic"):
    a = torch.as_tensor(initial, dtype=torch.float32, device=device)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    z = (a - torch.tensor(norm["initial_mean"], device=device)[None, :, None, None]) / torch.tensor(norm["initial_std"], device=device)[None, :, None, None]
    x = (x - torch.tensor(norm["x_mean"], device=device)) / torch.tensor(norm["x_std"], device=device)
    h, w = a.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=device) * 2 * torch.pi / h,
                            torch.arange(w, device=device) * 2 * torch.pi / w, indexing="ij")
    c = torch.stack([xx.sin(), xx.cos(), yy.sin(), yy.cos()] if coordinates == "periodic" else [xx, yy])
    return torch.cat([z, x[:, :, None, None].expand(-1, -1, h, w), c[None].expand(len(x), -1, -1, -1)], 1)


def decode(z, norm):
    return z * torch.tensor(norm["target_std"], device=z.device)[None, :, None, None] + torch.tensor(norm["target_mean"], device=z.device)[None, :, None, None]


def training_loss(z, truth, norm, recipe):
    pred = decode(z, norm)
    if recipe["loss"] == "relative":
        denominator = truth.square().mean((-2, -1)).clamp_min(recipe["field_floor"] ** 2)
        field = ((pred - truth).square().mean((-2, -1)) / denominator).mean()
    else:
        st = torch.tensor(norm["target_std"], device=z.device)[None, :, None, None]
        field = ((pred - truth) / st).square().mean()
    scale = torch.tensor(norm["outcome_scale"], device=z.device)
    outcome = ((physical_outcomes(pred) - physical_outcomes(truth)) / scale).square().mean()
    return field + 0.1 * outcome


def predict_member(model, x, norm, device, return_fields=False):
    """Per-row prediction exactly as predict_member7: analytic initial fields, encode, decode, read outcomes."""
    model.eval()
    values, fields = [], []
    with torch.no_grad():
        for row in np.asarray(x, float):
            ini = initial_fields(row[None])
            out = decode(model(encode(ini, row[None], norm, device)), norm)
            values.append(physical_outcomes(out)[0].cpu().numpy())
            if return_fields:
                fields.append(out[0].cpu().numpy())
    values = np.asarray(values).reshape(-1, len(C.OUTCOMES))
    return (values, np.asarray(fields)) if return_fields else values


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def tensor_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}


def count_real(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def load_member(dest, device="cpu", expected_sha256=None):
    dest = Path(dest)
    weights = dest / "weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights are not present at {dest.name}; refit the study version.")
    if expected_sha256 and digest(weights) != expected_sha256:
        raise ValueError(f"Model weights at {dest.name} were modified; refusing to load.")
    contract = json.loads((dest / "contract.json").read_text())
    model = TransferFNO(len(C.CONTROLS), contract["architecture"], contract["recipe"]["coordinates"]).to(device)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location=device), strict=True)
    return model, contract


# --------------------------------------------------------------------------- fitting
@dataclass
class FitData:
    """Train/development arrays from one immutable study version. No calibration or verification data."""
    train_ids: list
    train_x: np.ndarray
    train_initial: np.ndarray
    train_result: np.ndarray
    dev_x: np.ndarray
    dev_y: np.ndarray

    @property
    def train_y(self):
        return physical_outcomes(self.train_result)


def fit_member(data: FitData, recipe: dict, seed: int, dest, source_state, device="cpu",
               queries: dict | None = None, progress=None, should_cancel=None):
    """Adapt one ensemble member (fit_member7 semantics). Writes weights.pt, contract.json,
    learning.csv, predictions.npz (query roles) and result.json in dest.

    queries: role -> controls [n, 3], predicted after training with the selected checkpoint.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    architecture = recipe["architecture"]
    started = time.perf_counter()
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    x, ini, f, y = data.train_x, data.train_initial, data.train_result, data.train_y
    n = normalize_training(x, ini, f, y, data.train_ids)
    dx, dy = data.dev_x, data.dev_y
    model = TransferFNO(len(C.CONTROLS), architecture, recipe["coordinates"])
    model.backbone.load_state_dict(source_state, strict=True)
    model.to(device)
    identity = {"source_sha256": recipe["source_sha256"], "real_parameters": count_real(model),
                "train_ids": list(data.train_ids), "member_seed": seed}
    encoded = encode(ini, x, n, device, recipe["coordinates"])
    truth = torch.as_tensor(f, device=device)
    opt = torch.optim.Adam([{"params": model.conditioner.parameters(), "lr": recipe["adapter_lr"]},
                            {"params": model.backbone.parameters(), "lr": recipe["lr"]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, recipe["steps"])
    rng = np.random.default_rng(seed)
    best, best_step, trace, complete_steps, cap = float("inf"), None, [], 0, False
    loss = None
    fit_start = time.perf_counter()
    for step in range(recipe["steps"] + 1):
        if step:
            sync(device)
            if time.perf_counter() - fit_start >= recipe["max_seconds"]:
                cap = True
                break
            if should_cancel and step % 10 == 0 and should_cancel():
                raise Cancelled("Fit cancelled")
            model.train()
            opt.zero_grad(set_to_none=True)
            b = rng.choice(len(x), min(len(x), recipe["batch"]), replace=False)
            loss = training_loss(model(encoded[b]), truth[b], n, recipe)
            if not torch.isfinite(loss):
                raise MemberIncomplete("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise MemberIncomplete("Nonfinite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            complete_steps = step
        if step % recipe["validate_every"] == 0 or step == recipe["steps"]:
            vp = predict_member(model, dx, n, device)
            score = float(np.mean(((vp - dy) / n["outcome_scale"]) ** 2))
            if not np.isfinite(score):
                raise MemberIncomplete("Nonfinite development score")
            trace.append({"step": step, "development_score": score,
                          "training_sample_loss": float(loss.detach().cpu()) if step else None})
            if score < best:
                best = score
                torch.save(tensor_state(model), dest / "weights.tmp")
                (dest / "weights.tmp").replace(dest / "weights.pt")
                best_step = step
            if progress:
                progress({"step": step, "steps": recipe["steps"], "development_score": score,
                          "best_development_score": best, "best_step": best_step})
    with (dest / "learning.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["step", "development_score", "training_sample_loss"])
        writer.writeheader()
        writer.writerows(trace)
    write_json(dest / "contract.json", {"normalization": n, "recipe": recipe, "architecture": architecture,
                                        "identity": identity, "seed": seed})
    if cap or complete_steps != recipe["steps"]:
        raise MemberIncomplete(f"Member stopped at step {complete_steps} of {recipe['steps']} "
                               f"(safety cap {recipe['max_seconds']} s); incomplete members never enter a ready model.")
    model.load_state_dict(torch.load(dest / "weights.pt", weights_only=True, map_location=device), strict=True)
    roles = list((queries or {}).keys())
    qx = np.concatenate([np.asarray(queries[r], float).reshape(-1, 3) for r in roles]) if roles else np.zeros((0, 3))
    predictions = predict_member(model, qx, n, device) if len(qx) else np.zeros((0, 2))
    restored = TransferFNO(len(C.CONTROLS), architecture, recipe["coordinates"]).to(device)
    restored.load_state_dict(torch.load(dest / "weights.pt", weights_only=True, map_location=device), strict=True)
    check_x = qx[:2] if len(qx) >= 2 else dx[:2]
    reference = predictions[:2] if len(qx) >= 2 else predict_member(model, check_x, n, device)
    probe = predict_member(restored, check_x, n, device)
    if not np.allclose(probe, reference, atol=1e-06, rtol=1e-05):
        raise MemberIncomplete("Saved member reload mismatch")
    batched = _predict_batched(restored, check_x, n, device)
    if not np.allclose(probe, batched, atol=1e-06, rtol=1e-05):
        raise MemberIncomplete("Query batch invariance failed")
    counts = [len(np.asarray(queries[r]).reshape(-1, 3)) for r in roles]
    save_npz(dest / "predictions.npz", roles=np.array(sum([[r] * c for r, c in zip(roles, counts)], []), dtype=str),
             values=predictions)
    result = {"seconds": time.perf_counter() - started, "fit_seconds": time.perf_counter() - fit_start,
              "completed_steps": complete_steps, "time_cap_hit": cap,
              "best_step": best_step, "best_development_score": best, "reload_matches": True, "query_batch_invariant": True,
              "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
              "weights_bytes": (dest / "weights.pt").stat().st_size, "weights_sha256": digest(dest / "weights.pt"),
              "identity": identity, "device": device}
    write_json(dest / "result.json", result)
    del model, restored, opt, sched, encoded, truth
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _predict_batched(model, x, norm, device):
    """One batched forward over all rows; used to check batch invariance against per-row prediction."""
    model.eval()
    with torch.no_grad():
        out = decode(model(encode(initial_fields(x), x, norm, device)), norm)
    return physical_outcomes(out).cpu().numpy()


def predict_serving(model, x, norm, device, return_fields=False, chunk=4):
    """Chunked batched prediction for interactive serving (Explore/Next runs). Chunks of 4 were fastest on CPU.

    Fitting, checkpoint selection, calibration and recorded predictions keep the exact per-row
    predict_member path; every member checks that batched and per-row outputs agree within 1e-6.
    """
    model.eval()
    x = np.asarray(x, float).reshape(-1, len(C.CONTROLS))
    values, fields = [], []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            part = x[i:i + chunk]
            out = decode(model(encode(initial_fields(part), part, norm, device)), norm)
            values.append(physical_outcomes(out).cpu().numpy())
            if return_fields:
                fields.append(out.cpu().numpy())
    values = np.concatenate(values).reshape(-1, len(C.OUTCOMES))
    return (values, np.concatenate(fields)) if return_fields else values


# --------------------------------------------------------------------------- uncertainty
def conformal_quantile(scores, level):
    s = np.asarray(scores, float)
    if s.ndim != 1 or not len(s) or (not np.isfinite(s).all()) or np.any(s < 0) or (not 0 < level < 1):
        raise ValueError("Conformal scores must be a finite nonnegative vector and level in (0,1)")
    rank = int(np.ceil((len(s) + 1) * level - 1e-12))
    if rank > len(s):
        raise ValueError("Not enough calibration settings for a finite interval at this level")
    return {"q": float(np.partition(s, rank - 1)[rank - 1]), "rank": rank, "n": len(s)}


def fit_rules(truth, mean, sd, floor, levels):
    truth, mean, sd, floor = (np.asarray(v, float) for v in (truth, mean, sd, floor))
    if truth.shape != mean.shape or sd.shape != mean.shape or mean.ndim != 2 or (mean.shape[1] != 2):
        raise ValueError("Expected matched [settings,2] outcomes")
    if not all(np.isfinite(v).all() for v in [truth, mean, sd, floor]) or np.any(sd < 0) or floor.shape != (2,) or np.any(floor <= 0):
        raise ValueError("Nonfinite predictions or invalid scales")
    scale = np.maximum(sd, floor)
    resid = np.abs(truth - mean)
    rules = {}
    for level in levels:
        marginal = [conformal_quantile(resid[:, j] / scale[:, j], level) for j in range(2)]
        joint = conformal_quantile((resid / scale).max(1), level)
        constant = conformal_quantile((resid / floor).max(1), level)
        rules[str(level)] = {"marginal": marginal, "joint": joint, "constant_joint": constant}
    return rules


def interval(mean, sd, floor, rule, level, kind):
    from scipy.stats import norm as normal_dist

    mean, sd, floor = (np.asarray(v, float) for v in (mean, sd, floor))
    scale = np.maximum(sd, floor)
    if kind == "raw":
        half = normal_dist.ppf((1 + level) / 2) * sd
    elif kind == "marginal":
        half = scale * np.array([q["q"] for q in rule["marginal"]])
    elif kind == "joint":
        half = scale * rule["joint"]["q"]
    elif kind == "constant_joint":
        half = np.broadcast_to(floor * rule["constant_joint"]["q"], mean.shape)
    else:
        raise ValueError("Unknown interval rule")
    return (mean - half, mean + half)


def wilson(hits, n):
    from scipy.stats import norm as normal_dist

    if n <= 0:
        raise ValueError("Coverage requires settings")
    z = normal_dist.ppf(0.975)
    a = hits / n
    den = 1 + z * z / n
    center = (a + z * z / (2 * n)) / den
    half = z * math.sqrt(a * (1 - a) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def outcome_floor(norm, recipe):
    """Training-only scale floor: max(scale_fraction * train outcome SD, absolute floor)."""
    return np.maximum(np.array(norm["outcome_scale"]) * recipe["scale_fraction"], recipe["scale_absolute_floor"])


def ensemble(member_values):
    members = np.stack(member_values)
    if len(members) < 2:
        raise ValueError("An ensemble spread needs at least two members")
    return members.mean(0), members.std(0, ddof=1)


def readiness(probe_mean, probe_sd, floor, rule, tolerance, level):
    """Interval-width criterion on label-free probes (Task 8 readiness8). A heuristic, not a guarantee."""
    lo, hi = interval(probe_mean, probe_sd, floor, rule, level, "joint")
    ratios = (hi - lo) / 2 / np.array(tolerance)
    return {"probe_ready_fraction": float((ratios <= 1).all(1).mean()), "probe_max_halfwidth_ratio": float(ratios.max()),
            "criterion_met": bool((ratios <= 1).all(1).mean() >= 0.9 and ratios.max() <= 2.0), "probes": int(len(ratios))}


def verification_metrics(truth, mean, sd, floor, rule, tolerance, level):
    """Study-specific benchmark metrics on verification runs (never used for fitting or calibration)."""
    tolerance = np.array(tolerance)
    ratio = np.abs(truth - mean) / tolerance
    success = (ratio <= 1).all(1)
    out = {"n": int(len(truth)), "joint_accuracy_success": float(success.mean()),
           "joint_accuracy_ci": list(wilson(int(success.sum()), len(success))),
           "max_error_tolerance_ratio": float(ratio.max()),
           "rmse": {o: float(np.sqrt(np.mean((truth[:, j] - mean[:, j]) ** 2))) for j, o in enumerate(C.OUTCOME_NAMES)},
           "tolerance": tolerance.tolist(), "metric": "absolute nondimensional outcome error"}
    if rule is not None:
        lo, hi = interval(mean, sd, floor, rule, level, "joint")
        covered = ((truth >= lo) & (truth <= hi)).all(1)
        out["joint_coverage"] = float(covered.mean())
        out["joint_coverage_ci"] = list(wilson(int(covered.sum()), len(covered)))
    return out
