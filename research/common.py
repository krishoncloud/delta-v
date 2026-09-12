"""Local-only evaluation utilities. No downloads or implicit dataset discovery."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deltav_core as core

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def trajectory_key(record):
    return json.dumps([record[k] for k in ("system", "source_revision", "source_file", "trajectory")], separators=(",", ":"))

def read_manifest(path, checkpoint=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    records = manifest["windows"]
    if not records:
        raise ValueError("No prepared windows in the manifest")
    groups = {}
    seen_windows = set()
    for r in records:
        if r["system"] not in core.DATASETS or r["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown system or split")
        if not r["source_revision"] or not r["source_file"] or "start_index" not in r:
            raise ValueError("Every window needs a pinned source and start index")
        key = trajectory_key(r)
        if key in groups and groups[key] != r["split"]:
            raise ValueError("Trajectory leakage across splits: " + key)
        groups[key] = r["split"]
        window = (key, r["start_index"])
        if window in seen_windows:
            raise ValueError("Duplicate window")
        seen_windows.add(window)
        r["local_path"] = (path.parent / r["path"]).resolve()
        if sha256(r["local_path"]) != r["sha256"]:
            raise ValueError("Prepared window hash mismatch")
    ck = manifest.get("checkpoint", {})
    if not ck.get("exposure_complete") or not ck.get("provenance_note"):
        raise ValueError("Recover complete training/validation/normalization exposure records and cite the training notebook before evaluation")
    exposed = set(ck["exposed_trajectories"])
    test_keys = {k for k, split in groups.items() if split == "test"}
    if test_keys & exposed:
        raise ValueError("Test trajectories were exposed to the checkpoint")
    if checkpoint and sha256(checkpoint) != ck["sha256"]:
        raise ValueError("Checkpoint hash does not match the provenance manifest")
    for system in {r["system"] for r in records}:
        spec = manifest["statistics"][system]
        if not spec.get("provenance_verified"):
            raise ValueError("Statistics provenance must be verified for " + system)
        if set(spec["fit_trajectories"]) & {k for k,v in groups.items() if v != "train"}:
            raise ValueError("Statistics must not use validation/test trajectories")
        spec["local_path"] = (path.parent / spec["path"]).resolve()
        if sha256(spec["local_path"]) != spec["sha256"]:
            raise ValueError("Statistics hash mismatch")
    return manifest

def stats_for(manifest, system):
    with np.load(manifest["statistics"][system]["local_path"], allow_pickle=False) as z:
        return {"mean": z["mean"], "std": z["std"]}

def physical_frames(record):
    arr = np.load(record["local_path"], allow_pickle=False)
    if arr.ndim != 4 or arr.shape[0] != 5 or arr.shape[-1] != core.DATASETS[record["system"]]["n_fields"] or not np.isfinite(arr).all():
        raise ValueError("Expected five finite native-field snapshots")
    frames = []
    for raw in arr:
        field, _ = core.build_canonical(raw.astype(np.float32), record["system"])
        if core.DATASETS[record["system"]].get("derive_velocity"):
            field = core.derive_velocity_inplace(field)
        frames.append(core.resample(torch.from_numpy(field)))
    return frames

def fit_training_stats(records):
    """Fit only this data budget, never validation/test; weight window pixels equally."""
    total = torch.zeros(4, dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for record in records:
        if record["split"] != "train":
            raise ValueError("Normalization accepts training windows only")
        for frame in physical_frames(record):
            values = frame.double().reshape(4, -1)
            total += values.sum(1)
            squares += values.square().sum(1)
            count += values.shape[1]
    if not count:
        raise ValueError("No training pixels for normalization")
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt()
    return {"mean": mean.float().numpy(), "std": std.float().numpy()}

def tensors(record, stats):
    frames = [core.normalize(frame, stats) for frame in physical_frames(record)]
    return torch.cat(frames[:4]), frames[4]

def evaluate(model, records, manifest, device, statistics_override=None):
    from main import _quality_metrics
    result = []
    model.eval()
    with torch.no_grad():
        for r in records:
            stats = statistics_override if statistics_override is not None else stats_for(manifest, r["system"])
            x, y = tensors(r, stats)
            prediction = model(x.unsqueeze(0).to(device))[0].cpu()
            pred = core.denormalize(prediction, stats).numpy()
            truth = core.denormalize(y, stats).numpy()
            result.append({"system": r["system"], "trajectory": trajectory_key(r), "start_index": r["start_index"], "quality": _quality_metrics(pred, truth)})
    # Equal trajectory weight prevents trajectories with more windows dominating.
    trajectories = {}
    for r in result:
        trajectories.setdefault((r["system"], r["trajectory"]), []).append(r)
    means = {}
    for (system, _), rows in trajectories.items():
        means.setdefault(system, []).append({c: {m: float(np.mean([r["quality"][c][m] for r in rows if r["quality"][c][m] is not None])) if any(r["quality"][c][m] is not None for r in rows) else None for m in ("rel_l2", "rmse", "ssim")} for c in core.CANONICAL})
    aggregate = {s: {c: {m: float(np.mean([r[c][m] for r in rows if r[c][m] is not None])) if any(r[c][m] is not None for r in rows) else None for m in ("rel_l2", "rmse", "ssim")} for c in core.CANONICAL} for s, rows in means.items()}
    return {"per_window": result, "equal_trajectory_mean_by_system": aggregate}
