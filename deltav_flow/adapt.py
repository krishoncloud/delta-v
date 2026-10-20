"""Adapt one immutable study version into an immutable ensemble model, and serve it.

Role discipline (prompt section 6): train fits and normalizes; development only
selects checkpoints; calibration only fits the conformal rule; verification is
scored after the model is sealed. Candidates and probes are controls only.
"""
from __future__ import annotations

import csv
import json
import platform
import shutil
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from . import contract as C
from . import engine as E

PROFILES = {
    "task8-fno-lhs-v1": {"label": "Standard fine-tuning recipe", "test_only": False},
    "real-source-smoke": {"label": "SMOKE: real pinned source, truncated training (60 steps). Not the recipe; not evidence.",
                          "test_only": True, "overrides": {"steps": 60, "validate_every": 20}},
    "machinery-tiny": {"label": "MACHINERY FIXTURE: tiny randomly initialised network, not the pretrained FNO. Plumbing only.",
                       "test_only": True,
                       "overrides": {"steps": 6, "validate_every": 3,
                                     "architecture": {"n_modes": [4, 4], "hidden_channels": 8, "in_channels": 16,
                                                      "out_channels": 4, "n_layers": 1}}},
}
DEFAULT_PROBE_SEED = 20261019


def recipe_for(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown fit profile {profile!r}")
    recipe = json.loads(json.dumps(C.RECIPE))
    recipe.update(PROFILES[profile].get("overrides", {}))
    recipe["profile"] = profile
    recipe["profile_label"] = PROFILES[profile]["label"]
    if profile == "machinery-tiny":
        recipe["source_sha256"] = "machinery-fixture"
    return recipe


def machinery_source_state(architecture):
    """A seeded random tiny backbone for plumbing tests. Never the pretrained source."""
    import torch
    from neuralop.models import FNO

    torch.manual_seed(0)
    return {k: v.detach().clone() for k, v in FNO(**architecture).state_dict().items() if isinstance(v, torch.Tensor)}


def default_probes(ranges, n=C.DEFAULT_PROBES, seed=DEFAULT_PROBE_SEED):
    unit = np.random.default_rng(seed).uniform(0.01, 0.99, (n, 3))
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    return [f"probe-{i:04d}" for i in range(n)], lo + unit * (hi - lo)


def cpu_name():
    import subprocess

    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() + " CPU"
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{platform.machine()} CPU"


def environment(device):
    import torch

    names = ["torch", "neuraloperator", "numpy", "scipy"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: metadata.version(n) for n in names},
            "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else cpu_name(),
            "cuda": torch.version.cuda, "precision": "float32 model, float64 solver"}


def study_roles(study):
    """Index arrays per role from a ValidatedStudy (or combined version)."""
    roles = np.array([r["role"] for r in study.runs])
    return {role: np.flatnonzero(roles == role) for role in C.ROLES}


def fit_data(study) -> E.FitData:
    idx = study_roles(study)
    x = study.x()
    t, d = idx["train"], idx["development"]
    # Column-major like the notebook's pandas .to_numpy(), so normalizer reductions match to the last bit.
    return E.FitData([study.runs[i]["run_id"] for i in t], np.asfortranarray(x[t]), study.initial[t], study.result[t],
                     np.asfortranarray(x[d]), study.outcomes[d])


def adapt_study(study, version_key: str, profile: str, model_dir, source_state, device="cpu",
                allow_uncalibrated=False, progress=None, should_cancel=None, member_dir=None):
    """Fit every member, aggregate, calibrate, score verification and seal model.json.

    Verified complete members in member_dir (same signature) are reused after interruption.
    Partially trained members restart from their seed; optimizer-step resume is not implemented.
    """
    model_dir = Path(model_dir)
    member_dir = Path(member_dir or model_dir)
    recipe = recipe_for(profile)
    started = time.perf_counter()
    idx = study_roles(study)
    x = study.x()
    n_cal = len(idx["calibration"])
    calibrated = n_cal >= C.CALIBRATED_MINIMUM
    if not calibrated and not allow_uncalibrated:
        raise ValueError(f"{n_cal} calibration runs; calibrated intervals need {C.CALIBRATED_MINIMUM}. "
                         "Add calibration runs or explicitly request a prediction-only fit.")
    ranges = [c["range"] for c in study.meta["controls"]]
    if study.probes is not None:
        probe_ids, probe_x = study.probes.ids, study.probes.x
    else:
        probe_ids, probe_x = default_probes(ranges)
    queries = {"calibration": x[idx["calibration"]], "verification": x[idx["verification"]], "probe": probe_x}
    if study.candidates is not None:
        queries["candidate"] = study.candidates.x
    data = fit_data(study)
    seeds = C.member_seeds(study.meta.get("seed", 0), recipe["member_base_seeds"])
    members, member_values, norms, timings = [], [], [], []
    for i, seed in enumerate(seeds):
        dest = member_dir / f"member_{seed}"
        signature = E.canonical({"recipe": recipe, "version": version_key, "seed": seed, "train_ids": data.train_ids,
                                 "queries": {k: np.asarray(v).round(12).tolist() for k, v in queries.items()}})
        record = _verified_member(dest, signature)
        reused = record is not None
        if not reused:
            if dest.exists():
                shutil.rmtree(dest)
            report = (lambda info, i=i, seed=seed: progress({"stage": "training", "member": i + 1, "members": len(seeds),
                                                             "member_seed": seed, "detail": f"member seed {seed}", **info})) if progress else None
            result = E.fit_member(data, recipe, seed, dest, source_state, device, queries, report, should_cancel)
            record = {"signature": signature, "weights_sha256": result["weights_sha256"],
                      "files": {name: E.digest(dest / name) for name in ("contract.json", "result.json", "predictions.npz", "learning.csv")}}
            E.write_json(dest / "complete.json", record)
        result = json.loads((dest / "result.json").read_text())
        contract = json.loads((dest / "contract.json").read_text())
        pred = E.load_npz(dest / "predictions.npz")
        member_values.append(pred["values"])
        norms.append(contract["normalization"])
        members.append({"seed": seed, "dir": dest.name, "weights_sha256": record["weights_sha256"],
                        "weights_bytes": result["weights_bytes"], "best_step": result["best_step"],
                        "best_development_score": result["best_development_score"], "seconds": result["seconds"],
                        "completed_steps": result["completed_steps"], "reused_after_interruption": reused,
                        "reload_matches": result["reload_matches"], "query_batch_invariant": result["query_batch_invariant"]})
        timings.append(result["seconds"])
        if progress:
            progress({"stage": "training", "member": i + 1, "members": len(seeds), "member_seed": seed,
                      "member_complete": True, "step": recipe["steps"], "steps": recipe["steps"]})
    if any(n != norms[0] for n in norms):
        raise E.MemberIncomplete("Members differ in supplied runs or normalizers")
    if progress:
        progress({"stage": "calibrating" if calibrated else "validating"})
    roles = E.load_npz(member_dir / members[0]["dir"] / "predictions.npz")["roles"]
    mean, sd = E.ensemble(member_values)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise E.MemberIncomplete("Nonfinite ensemble outputs")
    norm = norms[0]
    floor = E.outcome_floor(norm, recipe)
    at = {role: roles == role for role in queries}
    rule = None
    calibration = {"status": "uncalibrated", "n": n_cal, "reason": f"{n_cal} calibration runs < {C.CALIBRATED_MINIMUM}"}
    if calibrated:
        truth = study.outcomes[idx["calibration"]]
        rule = E.fit_rules(truth, mean[at["calibration"]], sd[at["calibration"]], floor, [recipe["level"]])[str(recipe["level"])]
        calibration = {"status": "calibrated", "method": "normalized joint split conformal", "level": recipe["level"],
                       "n": n_cal, "rule": rule, "floor": floor.tolist(),
                       "scope": "Marginal over iid future queries like the calibration runs. Not a per-query guarantee, "
                                "not conditional on a region, and recommendation locations or repeated calibration-based "
                                "stopping do not inherit it."}
    tolerance = study.meta.get("tolerance", C.DEFAULT_TOLERANCE)
    ready = E.readiness(mean[at["probe"]], sd[at["probe"]], floor, rule, tolerance, recipe["level"]) if rule else None
    verification = None
    if len(idx["verification"]):
        verification = E.verification_metrics(study.outcomes[idx["verification"]], mean[at["verification"]],
                                              sd[at["verification"]], floor, rule, tolerance, recipe["level"])
    E.save_npz(model_dir / "aggregate_predictions.npz", roles=roles, mean=mean, sd=sd, floor=floor,
               ids=np.array([study.runs[i]["run_id"] for i in idx["calibration"]]
                            + [study.runs[i]["run_id"] for i in idx["verification"]]
                            + list(probe_ids) + (list(study.candidates.ids) if study.candidates is not None else [])))
    model = {"recipe": recipe, "profile": profile, "profile_label": recipe["profile_label"],
             "version_key": version_key, "normalization": norm, "floor": floor.tolist(), "members": members,
             "member_seeds": seeds, "calibration": calibration, "readiness": ready, "verification": verification,
             "tolerance": tolerance, "ranges": ranges, "environment": environment(device),
             "role_counts_used": {"train": len(idx["train"]), "development": len(idx["development"]),
                                  "calibration": n_cal if calibrated else 0},
             "timing": {"members_seconds": timings, "total_seconds": time.perf_counter() - started}}
    model["aggregate_sha256"] = E.digest(model_dir / "aggregate_predictions.npz")
    E.write_json(model_dir / "model.json", model)
    return model


def model_result(folder):
    """The payload a worker reports for a sealed model: metadata, recorded query predictions, learning curves."""
    folder = Path(folder)
    meta = json.loads((folder / "model.json").read_text())
    stored = E.load_npz(folder / "aggregate_predictions.npz")
    if E.digest(folder / "aggregate_predictions.npz") != meta["aggregate_sha256"]:
        raise ValueError("Recorded predictions changed after sealing")
    learning = {}
    for m in meta["members"]:
        with (folder / m["dir"] / "learning.csv").open() as f:
            learning[str(m["seed"])] = [{k: (float(v) if v not in ("", None) else None) for k, v in row.items()}
                                        for row in csv.DictReader(f)]
    floor = np.array(meta["floor"])
    rule = meta["calibration"].get("rule")
    rows = {}
    for role in ("verification", "candidate", "probe"):
        mask = stored["roles"] == role
        if not mask.any():
            continue
        mean, sd = stored["mean"][mask], stored["sd"][mask]
        entry = {"ids": stored["ids"][mask].tolist(), "mean": mean.tolist(), "sd": sd.tolist()}
        if rule:
            lo, hi = E.interval(mean, sd, floor, rule, meta["recipe"]["level"], "joint")
            entry.update(lower=lo.tolist(), upper=hi.tolist())
        rows[role] = entry
    return {"model": meta, "predictions": rows, "learning": learning}


def _verified_member(dest: Path, signature: str):
    path = dest / "complete.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
        if record["signature"] != signature or E.digest(dest / "weights.pt") != record["weights_sha256"]:
            return None
        if any(E.digest(dest / name) != sha for name, sha in record["files"].items()):
            return None
        return record
    except (OSError, KeyError, ValueError):
        return None


class Ensemble:
    """Loaded, hash-verified ensemble for bounded inference."""

    def __init__(self, model_dir, device="cpu", member_dir=None):
        self.dir = Path(model_dir)
        member_dir = Path(member_dir or model_dir)
        self.meta = json.loads((self.dir / "model.json").read_text())
        self.device = device
        self.members = []
        for m in self.meta["members"]:
            model, contract = E.load_member(member_dir / m["dir"], device, m["weights_sha256"])
            if contract["normalization"] != self.meta["normalization"]:
                raise ValueError("Member normalizer differs from the sealed model")
            self.members.append(model)

    def check(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Queries must be finite [n, 3] controls")
        lo = np.array([r[0] for r in self.meta["ranges"]])
        hi = np.array([r[1] for r in self.meta["ranges"]])
        outside = np.flatnonzero(((x < lo) | (x > hi)).any(1))
        if len(outside):
            raise ValueError(f"Query {int(outside[0])} lies outside the study's declared control box; extrapolation is not offered.")
        return x

    def predict(self, x, fields=False):
        x = self.check(x)
        values, field_values = [], []
        for model in self.members:
            out = E.predict_serving(model, x, self.meta["normalization"], self.device, return_fields=fields)
            if fields:
                values.append(out[0])
                field_values.append(out[1])
            else:
                values.append(out)
        mean, sd = E.ensemble(values)
        floor = np.array(self.meta["floor"])
        level = self.meta["recipe"]["level"]
        raw_lo, raw_hi = E.interval(mean, sd, floor, None, level, "raw")
        result = {"mean": mean, "sd": sd, "raw": (raw_lo, raw_hi), "members": np.stack(values)}
        rule = self.meta["calibration"].get("rule")
        if rule:
            result["calibrated"] = E.interval(mean, sd, floor, rule, level, "joint")
        if fields:
            result["fields"] = np.stack(field_values).mean(0)
        return result
