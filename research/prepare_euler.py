"""Prepare bounded Euler trajectory windows without downloading full HDF5 files."""
import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_ranges import BoundedRanges
from common import euler_trajectory_plan, sha256


REPO = "polymathic-ai/euler_multi_quadrants_openBC"
SOURCE_REVISION = "84e3eb65faf286533714295515caf9b99b4717a7"
SOURCES = [
    ("1.13", "data/train/euler_multi_quadrants_openBC_gamma_1.13_C3H8_16_chunk_0.hdf5"),
    ("1.22", "data/train/euler_multi_quadrants_openBC_gamma_1.22_C2H6_15_chunk_0.hdf5"),
    ("1.33", "data/train/euler_multi_quadrants_openBC_gamma_1.33_H2O_20_chunk_0.hdf5"),
    ("1.365", "data/train/euler_multi_quadrants_openBC_gamma_1.365_Dry_air_1000_chunk_0.hdf5"),
]


def save_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    notebook = p.add_mutually_exclusive_group(required=True)
    notebook.add_argument("--notebook")
    notebook.add_argument("--notebook-sha256")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source-revision", default=SOURCE_REVISION)
    p.add_argument("--gammas", nargs="+", choices=[gamma for gamma, _ in SOURCES], default=[gamma for gamma, _ in SOURCES])
    p.add_argument("--trajectories-per-gamma", type=int, default=16)
    p.add_argument("--start-indices", type=int, nargs="+", default=[0, 32, 64, 96])
    p.add_argument("--byte-budget-per-trajectory-mib", type=int, default=160)
    args = p.parse_args()
    if args.trajectories_per_gamma < 4 or args.trajectories_per_gamma > 40:
        p.error("--trajectories-per-gamma must be between 4 and 40")
    if len(set(args.start_indices)) != len(args.start_indices) or any(i < 0 or i + 5 > 101 for i in args.start_indices):
        p.error("Start indices must be unique and each five-frame window must fit within 101 frames")
    if len(set(args.gammas)) != len(args.gammas) or args.byte_budget_per_trajectory_mib < 1:
        p.error("Gammas must be unique and the byte budget must be positive")
    notebook_hash = sha256(args.notebook) if args.notebook else args.notebook_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", notebook_hash):
        p.error("Notebook SHA-256 must be 64 hexadecimal characters")

    root = Path(args.output_dir)
    selected_sources = [(gamma, path) for gamma, path in SOURCES if gamma in args.gammas]
    api = HfApi()
    revision = args.source_revision
    source_info = {item.rfilename: item for item in api.get_paths_info(REPO, [path for _, path in selected_sources], repo_type="dataset", revision=revision)}
    missing = {path for _, path in selected_sources} - source_info.keys()
    if missing:
        raise FileNotFoundError(f"Pinned Well source files do not exist at {revision}: {sorted(missing)}")
    manifest = {
        "kind": "Euler transfer dataset with fixed trajectory splits",
        "source_repo": REPO,
        "source_revision": revision,
        "selection": {
            "gammas": [gamma for gamma, _ in selected_sources],
            "trajectories_per_gamma": args.trajectories_per_gamma,
            "start_indices": args.start_indices,
            "policy": "Deterministic trajectory hash within gamma; 50% train, 25% validation, 25% test at the default count.",
        },
        "checkpoint": {
            "sha256": sha256(args.checkpoint),
            "exposure_complete": False,
            "excluded_systems": ["euler_multi_quadrants_openBC"],
            "provenance_note": f"Supplied training notebook SHA-256 {notebook_hash} uses only the three non-Euler systems for the epoch-59 joint checkpoint. This establishes intended Euler exclusion, not complete cryptographic reconstruction of all checkpoint exposures.",
            "exposed_trajectories": [],
        },
        "windows": [],
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if {key: previous[key] for key in ("source_repo", "source_revision", "selection", "checkpoint")} != {key: manifest[key] for key in ("source_repo", "source_revision", "selection", "checkpoint")}:
            raise ValueError("Existing preparation has different source, selection or checkpoint; use another output directory")
        manifest = previous
        for record in manifest["windows"]:
            target = root / record["path"]
            if not target.is_file() or sha256(target) != record["sha256"]:
                raise ValueError(f"Existing prepared window is missing or corrupt: {target}")
    else:
        root.mkdir(parents=True, exist_ok=False)
        save_manifest(manifest_path, manifest)
    completed = {(r["source_file"], r["trajectory"], r["start_index"]): r for r in manifest["windows"]}

    for gamma, source_file in selected_sources:
        info = source_info[source_file]
        for trajectory, split in euler_trajectory_plan(source_file, args.trajectories_per_gamma):
            missing = [start for start in args.start_indices if (source_file, trajectory, start) not in completed]
            if not missing:
                print(f"already prepared gamma={gamma} trajectory={trajectory}", flush=True)
                continue
            raw = BoundedRanges(
                hf_hub_url(REPO, source_file, repo_type="dataset", revision=revision),
                info.size,
                budget=args.byte_budget_per_trajectory_mib * 1024 * 1024,
            )
            with raw, h5py.File(raw, "r") as source:
                if source["t0_fields/density"].shape != (40, 101, 512, 512):
                    raise ValueError("Unexpected Euler source shape")
                for start in missing:
                    density = source["t0_fields/density"][trajectory, start : start + 5]
                    energy = source["t0_fields/energy"][trajectory, start : start + 5]
                    pressure = source["t0_fields/pressure"][trajectory, start : start + 5]
                    momentum = source["t1_fields/momentum"][trajectory, start : start + 5]
                    window = np.stack([density, energy, pressure, momentum[..., 0], momentum[..., 1]], axis=-1).astype(np.float32)
                    relative = Path("windows") / f"gamma-{gamma}-trajectory-{trajectory}-start-{start}.npy"
                    target = root / relative
                    target.parent.mkdir(exist_ok=True)
                    if target.exists():
                        target.unlink()  # Incomplete write from a prior interrupted attempt.
                    with target.open("xb") as handle:
                        np.save(handle, window, allow_pickle=False)
                    manifest["windows"].append({
                        "system": "euler_multi_quadrants_openBC",
                        "gamma": float(gamma),
                        "source_revision": revision,
                        "source_file": source_file,
                        "trajectory": trajectory,
                        "start_index": start,
                        "split": split,
                        "path": str(relative),
                        "sha256": sha256(target),
                    })
                    save_manifest(manifest_path, manifest)
            print(f"prepared gamma={gamma} trajectory={trajectory} split={split} range_bytes={raw.downloaded}", flush=True)

    counts = {split: len({(r["source_file"], r["trajectory"]) for r in manifest["windows"] if r["split"] == split}) for split in ("train", "validation", "test")}
    print("complete", manifest_path, counts)


if __name__ == "__main__":
    main()
