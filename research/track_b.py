"""Retrospective nine-setting cooling-time design-run benchmark.

This is an outcome-table study, not a trajectory-held-out or prospective
simulation-cost result. It never reads large source files or the slices tree.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


SYSTEMS = ("turbulent_radiative_layer_2D", "rayleigh_benard", "shear_flow")
OUTCOMES = ("mass_flux", "turbulent_velocity")
FOLDS = ((2, 5), (3, 6), (1, 4))  # internal design settings only; endpoints remain observable
SEEDS = (42, 123, 2026)
LENGTH_SCALE = 0.25
NOISE_VARIANCE = 0.01


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path, system):
    with np.load(path, allow_pickle=False) as z:
        points = np.asarray(z["points"], dtype=float)
        values = np.asarray(z["values"], dtype=float)
        metrics = tuple(str(x) for x in z["metric_names"])
        dials = tuple(str(x) for x in z["dial_names"])
    if (points.ndim != 2 or values.shape != (len(points), 2)
            or metrics != OUTCOMES or not np.isfinite(points).all()
            or not np.isfinite(values).all() or np.any(points <= 0)):
        raise ValueError(f"Invalid outcome table: {system}")
    expected = ("t_cool",) if system == SYSTEMS[0] else (
        ("rayleigh_number", "prandtl") if system == SYSTEMS[1] else ("reynolds", "schmidt"))
    if dials != expected or points.shape[1] != len(expected):
        raise ValueError(f"Unexpected design dials: {system}: {dials}")
    order = np.argsort(points[:, 0], kind="stable")
    return points[order], values[order]


def log_position(x, lo, hi):
    return (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))


def source_training_data(tables):
    """No radiative labels enter warm-start pretraining."""
    xs, ys = [], []
    for system in SYSTEMS[1:]:
        points, values = tables[system]
        u = log_position(points[:, 0], points[:, 0].min(), points[:, 0].max())
        scale = np.maximum(values.std(axis=0), 1e-6)
        xs.extend(u.tolist())
        ys.extend(((values - values.mean(axis=0)) / scale).tolist())
    return np.asarray(xs, dtype=np.float32)[:, None], np.asarray(ys, dtype=np.float32)


def network(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(),
                               torch.nn.Linear(16, 16), torch.nn.Tanh(),
                               torch.nn.Linear(16, 2))


def train_network(model, x, y, steps):
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(-1, 1))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        optimizer.step()
    return model


def gp_predict(x, y, query):
    """Fixed-hyperparameter GP; no test-label tuning or optimizer search."""
    x = np.asarray(x, dtype=float).reshape(-1)
    query = np.asarray(query, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float)
    mean, scale = y.mean(axis=0), np.maximum(y.std(axis=0), 1e-6)
    z = (y - mean) / scale
    kernel = np.exp(-0.5 * ((x[:, None] - x[None, :]) / LENGTH_SCALE) ** 2)
    kernel += NOISE_VARIANCE * np.eye(len(x))
    cross = np.exp(-0.5 * ((query[:, None] - x[None, :]) / LENGTH_SCALE) ** 2)
    chol = np.linalg.cholesky(kernel)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, z))
    latent_var = np.maximum(1 - np.sum(np.linalg.solve(chol, cross.T) ** 2, axis=0), 0)
    std = np.sqrt(latent_var + NOISE_VARIANCE)[:, None] * scale
    return cross @ alpha * scale + mean, np.broadcast_to(std, (len(query), 2))


def mlp_predict(x, y, query, seeds, steps, warm_states=None):
    x = np.asarray(x, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mean, scale = y.mean(axis=0), np.maximum(y.std(axis=0), 1e-6)
    predictions = []
    for seed in seeds:
        model = network(seed)
        if warm_states is not None:
            model.load_state_dict(warm_states[seed])
        train_network(model, x, (y - mean) / scale, steps)
        with torch.no_grad():
            predictions.append(model(torch.from_numpy(query[:, None])).numpy() * scale + mean)
    stack = np.stack(predictions)
    return stack.mean(axis=0), np.maximum(stack.std(axis=0, ddof=1), 1e-6)


def initial_indices(pool, u):
    midpoint = min(pool[1:-1], key=lambda i: (abs(u[i] - 0.5), i))
    return [pool[0], midpoint, pool[-1]]


def run(tables, target, steps=300, source_steps=500):
    if target <= 0 or steps < 1 or source_steps < 1:
        raise ValueError("Target and training steps must be positive")
    points, truth = tables[SYSTEMS[0]]
    if len(points) != 9 or len(np.unique(points[:, 0])) != 9:
        raise ValueError("Track B requires the audited nine distinct cooling-time settings")
    u = log_position(points[:, 0], points[:, 0].min(), points[:, 0].max())
    sx, sy = source_training_data(tables)
    warm_states = {}
    for seed in SEEDS:
        model = train_network(network(seed), sx, sy, source_steps)
        warm_states[seed] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    records = []
    for fold, heldout in enumerate(FOLDS, start=1):
        pool = [i for i in range(9) if i not in heldout]
        acquired = initial_indices(pool, u)
        scale = np.maximum(np.ptp(truth[pool], axis=0), 1e-6)  # metric only; not used for fitting/acquisition
        for budget in range(3, len(pool) + 1):
            x, y = u[acquired], truth[acquired]
            outputs = {"gp": gp_predict(x, y, u[list(heldout)]),
                       "scratch": mlp_predict(x, y, u[list(heldout)], SEEDS, steps),
                       "fluid_warm_start": mlp_predict(x, y, u[list(heldout)], SEEDS, steps, warm_states)}
            for method, (prediction, uncertainty) in outputs.items():
                error = prediction - truth[list(heldout)]
                normalized_rmse = np.sqrt(np.mean((error / scale) ** 2, axis=0))
                records.append({"fold": fold, "heldout_indices": list(heldout),
                                "acquired_indices": list(acquired), "runs": budget,
                                "method": method, "t_cool": points[list(heldout), 0].tolist(),
                                "predicted": prediction.tolist(),
                                "observed": truth[list(heldout)].tolist(),
                                "uncertainty_1sigma": uncertainty.tolist(),
                                "normalized_rmse": normalized_rmse.tolist(),
                                "meets_target": bool(np.max(normalized_rmse) <= target),
                                "within_1sigma": (np.abs(error) <= uncertainty).tolist(),
                                "within_2sigma": (np.abs(error) <= 2 * uncertainty).tolist()})
            if budget < len(pool):
                remaining = [i for i in pool if i not in acquired]
                _, sd = gp_predict(x, y, u[remaining])
                # Common, label-blind GP acquisition schedule for all arms.
                acquired_scale = np.maximum(np.ptp(y, axis=0), 1e-6)
                next_index = min(remaining, key=lambda i: (-float(np.mean(sd[remaining.index(i)] / acquired_scale)), i))
                acquired.append(next_index)
    summary = []
    for method in ("gp", "scratch", "fluid_warm_start"):
        first = []
        for fold in range(1, len(FOLDS) + 1):
            passing = [r["runs"] for r in records if r["method"] == method and r["fold"] == fold and r["meets_target"]]
            first.append(min(passing) if passing else None)
        final = [r for r in records if r["method"] == method and r["runs"] == 7]
        summary.append({"method": method, "first_runs_to_target_by_fold": first,
                        "runs_saved_from_nine_setting_sweep_by_fold": [9 - n if n is not None else None for n in first],
                        "folds_meeting_target": sum(n is not None for n in first),
                        "final_normalized_rmse_mean": np.mean([r["normalized_rmse"] for r in final], axis=0).tolist(),
                        "final_68pct_coverage": float(np.mean([r["within_1sigma"] for r in final])),
                        "final_95pct_coverage": float(np.mean([r["within_2sigma"] for r in final]))})
    return {"protocol": {"study": "retrospective parameter-setting holdout, not trajectory-held-out",
                         "outcomes": OUTCOMES, "test_folds": FOLDS, "ensemble_seeds": SEEDS,
                         "target_normalized_rmse_each_outcome": target,
                         "normalization": "range of seven non-test design outcomes per fold; evaluation only",
                         "fixed_sweep_runs": 9, "scratch_and_warm_start_steps": steps,
                         "source_pretraining_steps": source_steps,
                         "acquisition": "shared maximum fixed-GP uncertainty; excludes held-out settings",
                         "warm_start": "other two published fluid outcome tables; no radiative labels or FNO field features"},
            "records": records, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiative", required=True)
    parser.add_argument("--rayleigh", required=True)
    parser.add_argument("--shear", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--source-steps", type=int, default=500)
    args = parser.parse_args()
    files = dict(zip(SYSTEMS, (args.radiative, args.rayleigh, args.shear)))
    tables = {name: read_table(path, name) for name, path in files.items()}
    report = run(tables, args.target, args.steps, args.source_steps)
    report["protocol"]["input_sha256"] = {name: sha256(path) for name, path in files.items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Saved Track B:", output / "results.json")
    for item in report["summary"]:
        print(item["method"], "first runs:", item["first_runs_to_target_by_fold"],
              "coverage 68/95:", round(item["final_68pct_coverage"], 3),
              round(item["final_95pct_coverage"], 3))


if __name__ == "__main__":
    main()
