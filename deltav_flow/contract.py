"""Frozen capability, bundle schema, recipe and provenance constants.

Changing any scientific value here changes the recipe. Bump RECIPE['id'] and
record why; do not edit in place to make a result look better.
"""
from __future__ import annotations

import math

SCHEMA = "deltav-study-bundle/1"
CAPABILITY_ID = "deltav-fluid-periodic2d-v1"

# Single source for every helper extracted from research code.
PROVENANCE = {
    "notebook": "research/DeltaV_Task8_MoLab.py",
    "notebook_sha256": "ed95f31c11a1eb81d6dff83d5a440e6624b698005dca7ce9ff32556724d5bab5",
    "executed_copy": "research/task8/results/20261019T025823Z/notebook_source.py (byte-identical)",
    "task8_protocol_sha256": "df369db23d69771ad092d5413b07afac3a670f2d217ebc9697d6880bfc256424",
    "extracted": ["solve_flow", "initial7", "physical_outcomes", "normalize_training", "TransferFNO",
                  "encode", "decode", "training_loss", "load_source", "fit_member7", "predict_member7",
                  "conformal_quantile7", "fit_rules7", "interval7", "static_order8 (fno_lhs)"],
}

SOURCE = {
    "repo": "KrishMalik/deltav-fluid",
    "revision": "e25651ff92cf1d8f8818164a0277b8613e83d8b3",
    "filename": "checkpoints/domain_fluid.pt",
    "sha256": "2aefbf631c8ce81f5037646b08d6e16ece9e736cbfda1b37ad4c924e261e1674",
    "bytes": 858147645,
}

ARCHITECTURE = {"n_modes": [32, 32], "hidden_channels": 128, "in_channels": 16, "out_channels": 4}

GRID = (64, 64)
CHANNELS = ["density", "pressure", "u", "v"]
CONTROLS = [
    {"name": "viscosity", "range": [0.02, 0.08], "unit": "nondimensional"},
    {"name": "forcing", "range": [0.2, 1.2], "unit": "nondimensional"},
    {"name": "initial_mix", "range": [0.2, 0.8], "unit": "nondimensional"},
]
CONTROL_NAMES = [c["name"] for c in CONTROLS]
RANGES = [c["range"] for c in CONTROLS]
OUTCOMES = [
    {"name": "kinetic_energy", "kind": "mean_kinetic_energy", "unit": "nondimensional",
     "definition": "0.5 * spatial mean(u^2 + v^2) of the result field"},
    {"name": "pressure_rms", "kind": "pressure_rms", "unit": "nondimensional",
     "definition": "sqrt(spatial mean((p - spatial mean(p))^2)) of the result field"},
]
OUTCOME_NAMES = [o["name"] for o in OUTCOMES]
INITIAL_GENERATOR = "deltav-periodic-initial-v1"

# Everything a compatible study.json must state, compared exactly.
STUDY_REQUIREMENTS = {
    "lane": "fluid",
    "capability": CAPABILITY_ID,
    "grid": list(GRID),
    "geometry": "fixed periodic square [0,2pi]^2",
    "boundary_conditions": ["periodic", "periodic"],
    "units": "nondimensional",
    "time": {"initial": 0.0, "result": 2.0},
    "fields": {
        "channels": CHANNELS,
        "layout": "channel,y,x",
        "coordinates": "x_j = 2*pi*j/64, y_i = 2*pi*i/64; field[c, i, j]",
        "dtype": "float32",
        "format": "npy",
    },
    "initial_condition": {"representation": "supplied_fields", "generator": INITIAL_GENERATOR},
}

ROLES = ["train", "development", "calibration", "verification"]
# Fit roles never include verification. Verification is benchmark evaluation only.
FIT_ROLES = ["train", "development", "calibration"]

LIMITS = {
    "train": (8, 64),
    "development": (12, 64),
    "calibration": (0, 400),
    "verification": (0, 400),
    "runs_total": 1000,
    "candidates": 4096,
    "probes": 1024,
    "zip_bytes": 64 * 1024 * 1024,
    "unpacked_bytes": 320 * 1024 * 1024,
    "files": 2200,
    "compression_ratio": 50.0,
    "text_bytes": 4 * 1024 * 1024,
    "name_length": 200,
}
CALIBRATED_MINIMUM = 39
DEFAULT_TOLERANCE = [0.015, 0.015]
DEFAULT_PROBES = 128

RECIPE = {
    "id": "task8-fno-lhs-v1",
    "description": "Standard fine-tuning recipe: full fine-tuning from the pinned source, "
                   "three members, development checkpoint selection, joint split conformal calibration",
    "steps": 3000,
    "batch": 2,
    "lr": 1e-4,
    "adapter_lr": 1e-3,
    "validate_every": 100,
    "loss": "relative",
    "field_floor": 0.01,
    "coordinates": "periodic",
    "schedule": "cosine",
    "max_seconds": 3600.0,
    "member_base_seeds": [1103, 2207, 3301],
    "seed_transform": "(base + study_seed * 10007) % 2**31",
    "level": 0.9,
    "scale_fraction": 0.01,
    "scale_absolute_floor": 1e-6,
    "architecture": ARCHITECTURE,
    "source_sha256": SOURCE["sha256"],
}


def member_seeds(study_seed: int, base=RECIPE["member_base_seeds"]) -> list[int]:
    return [int((s + int(study_seed) * 10007) % 2**31) for s in base]


def in_range(values) -> bool:
    return all(math.isfinite(v) and lo <= v <= hi for v, (lo, hi) in zip(values, RANGES))


def capability() -> dict:
    """Public description served by GET /flow/capabilities."""
    return {
        "capability": CAPABILITY_ID,
        "schema": SCHEMA,
        "scope": "One generated 2D periodic forced incompressible flow family on a fixed 64x64 grid. "
                 "Not arbitrary fluids, geometry, 3D, customer CFD or OpenFOAM.",
        "study": STUDY_REQUIREMENTS,
        "controls": CONTROLS,
        "outcomes": OUTCOMES,
        "roles": {
            "train": {"min": LIMITS["train"][0], "max": LIMITS["train"][1], "use": "adaptation"},
            "development": {"min": LIMITS["development"][0], "max": LIMITS["development"][1],
                            "use": "checkpoint selection only"},
            "calibration": {"min_for_calibrated": CALIBRATED_MINIMUM, "max": LIMITS["calibration"][1],
                            "use": "joint split-conformal calibration only"},
            "verification": {"min": 0, "max": LIMITS["verification"][1],
                             "use": "benchmark evaluation after fitting; never used to fit, select or calibrate"},
        },
        "recipe": RECIPE,
        "source": SOURCE,
        "limits": {k: v for k, v in LIMITS.items() if not isinstance(v, tuple)},
        "query": "controls inside the declared box; initial fields from " + INITIAL_GENERATOR
                 + "; no future fields or labels are used",
    }
