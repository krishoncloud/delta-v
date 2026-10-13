"""Prepare bounded Euler trajectory windows without downloading full HDF5 files."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounded_ranges import BoundedRanges
from common import euler_trajectory_plan, sha256


REPO = "polymathic-ai/euler_multi_quadrants_openBC"
SOURCES = [
    ("1.13", "data/train/euler_multi_quadrants_openBC_gamma_1.13_C3H8_16_chunk_0.hdf5"),
    ("1.22", "data/train/euler_multi_quadrants_openBC_gamma_1.22_C3H8_16_chunk_0.hdf5"),
    ("1.33", "data/train/euler_multi_quadrants_openBC_gamma_1.33_C3H8_16_chunk_0.hdf5"),
    ("1.365", "data/train/euler_multi_quadrants_openBC_gamma_1.365_C3H8_16_chunk_0.hdf5"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--notebook", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--trajectories-per-gamma", type=int, default=16)
    p.add_argument("--start-indices", type=int, nargs="+", default=[0, 32, 64, 96])
    p.add_argument("--byte-budget-per-trajectory-mib", type=int, default=160)
    args = p.parse_args()
    if args.trajectories_per_gamma < 4 or args.trajectories_per_gamma > 40:
        p.error("--trajectories-per-gamma must be between 4 and 40")
    if len(set(args.start_indices)) != len(args.start_indices) or any(i < 0 or i + 5 > 101 for i in args.start_indices):
        p.error("Start indices must be unique and each five-frame window must fit within 101 frames")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    api = HfApi()
    revision = api.dataset_info(REPO).sha
    source_info = {item.rfilename: item for item in api.get_paths_info(REPO, [path for _, path in SOURCES], repo_type="dataset", revision=revision)}
    manifest = {
        "kind": "Euler transfer dataset with fixed trajectory splits",
        "source_repo": REPO,
        "source_revision": revision,
        "selection": {
            "gammas": [gamma for gamma, _ in SOURCES],
            "trajectories_per_gamma": args.trajectories_per_gamma,
            "start_indices": args.start_indices,
            "policy": "Deterministic trajectory hash within gamma; 50% train, 25% validation, 25% test at the default count.",
        },
        "checkpoint": {
            "sha256": sha256(args.checkpoint),
            "exposure_complete": False,
            "excluded_systems": ["euler_multi_quadrants_openBC"],
            "provenance_note": f"Supplied training notebook SHA-256 {sha256(args.notebook)} uses only the three non-Euler systems for the epoch-59 joint checkpoint. This establishes intended Euler exclusion, not complete cryptographic reconstruction of all checkpoint exposures.",
            "exposed_trajectories": [],
        },
        "windows": [],
    }

    for gamma, source_file in SOURCES:
        info = source_info[source_file]
        for trajectory, split in euler_trajectory_plan(source_file, args.trajectories_per_gamma):
            raw = BoundedRanges(
                hf_hub_url(REPO, source_file, repo_type="dataset", revision=revision),
                info.size,
                budget=args.byte_budget_per_trajectory_mib * 1024 * 1024,
            )
            with raw, h5py.File(raw, "r") as source:
                if source["t0_fields/density"].shape != (40, 101, 512, 512):
                    raise ValueError("Unexpected Euler source shape")
                for start in args.start_indices:
                    density = source["t0_fields/density"][trajectory, start : start + 5]
                    energy = source["t0_fields/energy"][trajectory, start : start + 5]
                    pressure = source["t0_fields/pressure"][trajectory, start : start + 5]
                    momentum = source["t1_fields/momentum"][trajectory, start : start + 5]
                    window = np.stack([density, energy, pressure, momentum[..., 0], momentum[..., 1]], axis=-1).astype(np.float32)
                    relative = Path("windows") / f"gamma-{gamma}-trajectory-{trajectory}-start-{start}.npy"
                    target = root / relative
                    target.parent.mkdir(exist_ok=True)
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
            print(f"prepared gamma={gamma} trajectory={trajectory} split={split} range_bytes={raw.downloaded}", flush=True)
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    counts = {split: len({(r["source_file"], r["trajectory"]) for r in manifest["windows"] if r["split"] == split}) for split in ("train", "validation", "test")}
    print("complete", root / "manifest.json", counts)


if __name__ == "__main__":
    main()
