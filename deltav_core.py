"""
Delta-V inference core.

Mirrors the notebook's canonical adapter, normalization, and parametric-tier
logic exactly (Steps 3/7/8/9 of DeltaV_FluidDomain_molab_v2.py), extracted to
plain Python so it can run outside marimo in a FastAPI process.

This module does NOT touch The Well / HDF5 at request time — the parametric
tables and FNO checkpoint are pulled once from HF at startup (see main.py),
and everything here operates on already-loaded arrays/tensors.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.interpolate import interp1d, griddata

# ---------------------------------------------------------------------------
# Canonical vocabulary — must match the notebook exactly.
# ---------------------------------------------------------------------------
CANONICAL = ["scalar", "pressure", "velocity_x", "velocity_y"]
C = len(CANONICAL)
TARGET_HW = (256, 256)
N_HISTORY = 4
SCALAR, PRES, VX, VY = 0, 1, 2, 3

# Model config — must match what was actually trained (Step 10).
FNO_CONFIG = dict(n_modes=(32, 32), hidden_channels=128,
                   in_channels=N_HISTORY * C, out_channels=C)

# Dataset registry — dials + field_map, same as the notebook's Step 3.
# gamma/euler kept here for completeness even though it's probe-only.
DATASETS = {
    "turbulent_radiative_layer_2D": {
        "grid": (128, 384), "n_fields": 4,
        "dials": [{"name": "t_cool", "idx": 0}],
        "field_map": {"scalar": 0, "pressure": 1, "velocity_x": 2, "velocity_y": 3},
    },
    "rayleigh_benard": {
        "grid": (512, 128), "n_fields": 4,
        "dials": [{"name": "rayleigh_number", "idx": 0}, {"name": "prandtl", "idx": 1}],
        "field_map": {"scalar": 0, "pressure": 1, "velocity_x": 2, "velocity_y": 3},
    },
    "shear_flow": {
        "grid": (256, 512), "n_fields": 4,
        "dials": [{"name": "reynolds", "idx": 0}, {"name": "schmidt", "idx": 1}],
        "field_map": {"scalar": 0, "pressure": 1, "velocity_x": 2, "velocity_y": 3},
    },
    "euler_multi_quadrants_openBC": {
        "grid": (512, 512), "n_fields": 5, "derive_velocity": True,
        "dials": [{"name": "gamma", "idx": 0}],
        "field_map": {"scalar": 0, "pressure": 2, "velocity_x": 3, "velocity_y": 4},
        "max_source_files": 4,
    },
}
EULER = "euler_multi_quadrants_openBC"
FLUID_FAMILY = [k for k in DATASETS if k != EULER]


# ---------------------------------------------------------------------------
# Canonical adapter — identical to notebook Step 7.
# ---------------------------------------------------------------------------
def build_canonical(frame_hwF: np.ndarray, name: str):
    fmap = DATASETS[name]["field_map"]
    H, W, _n = frame_hwF.shape
    out = np.zeros((C, H, W), dtype=np.float32)
    mask = np.zeros(C, dtype=np.float32)
    for ci, field in enumerate(CANONICAL):
        if field in fmap:
            out[ci] = frame_hwF[..., fmap[field]]
            mask[ci] = 1.0
    return out, mask


def resample(t_chw: torch.Tensor) -> torch.Tensor:
    return Fnn.interpolate(
        t_chw.unsqueeze(0), size=TARGET_HW, mode="bilinear", align_corners=False
    ).squeeze(0)


def derive_velocity_inplace(frame_chw: np.ndarray) -> np.ndarray:
    dens = frame_chw[SCALAR] + 1e-6
    frame_chw[VX] = frame_chw[VX] / dens
    frame_chw[VY] = frame_chw[VY] / dens
    return frame_chw


def normalize(t: torch.Tensor, stats: dict) -> torch.Tensor:
    m = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
    s = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)
    return (t - m) / (s + 1e-6)


def denormalize(t: torch.Tensor, stats: dict) -> torch.Tensor:
    m = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
    s = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)
    return t * (s + 1e-6) + m


# ---------------------------------------------------------------------------
# Stats loading — reads the .npz files pulled from HF (see main.py startup).
# ---------------------------------------------------------------------------
def load_stats(stats_dir: str, name: str) -> dict:
    path = os.path.join(stats_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stats file for '{name}' at {path}. "
            f"Pull checkpoints/stats from HF before starting the server."
        )
    z = np.load(path)
    return {"mean": z["mean"], "std": z["std"]}


# ---------------------------------------------------------------------------
# Parametric tier — identical logic to notebook Step 9, reading a
# precomputed table instead of scanning HDF5 at request time.
# ---------------------------------------------------------------------------
def load_param_table(ckpt_dir: str, name: str) -> dict:
    path = os.path.join(ckpt_dir, f"parametric_{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No parametric table for '{name}' at {path}. "
            f"Run build_param_table() in the notebook and push it to HF first."
        )
    z = np.load(path, allow_pickle=True)
    return {"name": name, "points": z["points"], "values": z["values"],
            "metrics": list(z["metric_names"]), "dials": list(z["dial_names"])}


def make_predictor(table: dict, fallback_nearest: bool = False):
    """Identical to notebook Step 9's make_predictor. Returns a callable with
    a `.range` attribute describing the valid dial domain."""
    pts, vals = table["points"], table["values"]
    ndial = pts.shape[1]
    mnames, dnames = table["metrics"], table["dials"]

    if ndial == 1:
        x = pts[:, 0]
        fns = {m: interp1d(x, vals[:, k]) for k, m in enumerate(mnames)}
        lo, hi = float(x.min()), float(x.max())

        def predict1(*args, **kw):
            v = float(kw.get(dnames[0], args[0] if args else None))
            if not (lo <= v <= hi):
                raise ValueError(f"{dnames[0]}={v} outside [{lo}, {hi}]")
            return {dnames[0]: v, **{m: float(f(v)) for m, f in fns.items()}}

        predict1.range = {dnames[0]: (lo, hi)}
        return predict1

    def predict2(*args, **kw):
        q = np.array([[float(kw[d]) if d in kw else float(args[j])
                       for j, d in enumerate(dnames)]])
        out = {d: float(q[0][j]) for j, d in enumerate(dnames)}
        for k, m in enumerate(mnames):
            r = griddata(pts, vals[:, k], q, method="linear")
            if np.isnan(r[0]):
                if not fallback_nearest:
                    raise ValueError(
                        f"query {out} outside the sampled "
                        f"({', '.join(dnames)}) convex hull; "
                        f"pass fallback_nearest=True to snap to the nearest point"
                    )
                r = griddata(pts, vals[:, k], q, method="nearest")
            out[m] = float(r[0])
        return out

    predict2.range = {d: (float(pts[:, j].min()), float(pts[:, j].max()))
                      for j, d in enumerate(dnames)}
    return predict2


# ---------------------------------------------------------------------------
# FNO model construction / loading.
# ---------------------------------------------------------------------------
def build_fno(device: str = "cpu"):
    """Constructs the FNO with the exact architecture the checkpoint was
    trained with. Import is local so the module can be imported for the
    parametric tier alone without neuralop installed, if ever needed."""
    from neuralop.models import FNO
    return FNO(**FNO_CONFIG).to(device)


def load_fno_checkpoint(model: "torch.nn.Module", ckpt_path: str, device: str = "cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("epoch")


# ---------------------------------------------------------------------------
# Full-field (deep-dive) inference.
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_fno_inference(
    model: "torch.nn.Module",
    history_frames: list[np.ndarray],  # list of N_HISTORY raw (H, W, n_fields) arrays
    name: str,
    stats: dict,
    device: str = "cpu",
) -> np.ndarray:
    """
    history_frames: N_HISTORY consecutive raw frames for `name`'s native
    field layout, oldest first. Returns the predicted next canonical frame
    as a (C, H, W) array in ORIGINAL (denormalized) units, resampled to
    TARGET_HW — matching what the model was trained to output.
    """
    if len(history_frames) != N_HISTORY:
        raise ValueError(f"expected {N_HISTORY} history frames, got {len(history_frames)}")

    derive = bool(DATASETS[name].get("derive_velocity"))
    xs = []
    for raw in history_frames:
        frame, _mask = build_canonical(raw, name)
        if derive:
            frame = derive_velocity_inplace(frame)
        t = resample(torch.tensor(frame))
        t = normalize(t, stats)
        xs.append(t)
    x = torch.cat(xs, dim=0).unsqueeze(0).to(device)  # (1, N_HISTORY*C, H, W)

    pred = model(x)[0]  # (C, H, W), normalized
    pred = denormalize(pred, stats)
    return pred.cpu().numpy()
