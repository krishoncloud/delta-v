"""Extract three bounded official-test reference clips; never accesses slices/."""
import argparse
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_url
from source_probe import BoundedRanges, SOURCES

def extract(item, root):
    system, revision, filename, start, scalar = item
    repo = "polymathic-ai/" + system
    path = "data/test/" + filename
    info = HfApi().get_paths_info(repo, [path], repo_type="dataset", revision=revision)[0]
    raw = BoundedRanges(hf_hub_url(repo, path, repo_type="dataset", revision=revision), info.size)
    with raw, h5py.File(raw, "r") as f:
        primary = f["t0_fields/" + scalar][0, start:start+11]
        pressure = f["t0_fields/pressure"][0, start:start+11]
        velocity = f["t1_fields/velocity"][0, start:start+11]
        arr = np.stack([primary, pressure, velocity[..., 0], velocity[..., 1]], axis=-1).astype(np.float32)
        time = f["dimensions/time"]
        times = (time[0, start:start+11] if time.ndim == 2 else time[start:start+11]).tolist()
    sid = "rollout-test-" + system
    np.save(root / (sid + ".npy"), arr, allow_pickle=False)
    dials = {"t_cool": .03} if system.startswith("turbulent") else {"rayleigh_number": 1e8, "prandtl": 1} if system == "rayleigh_benard" else {"reynolds": 1e5, "schmidt": 1}
    record = {"id": sid, "system": system, "dials": dials, "t_index": start, "times": times, "shape": list(arr.shape),
        "sha256": hashlib.sha256((root / (sid + ".npy")).read_bytes()).hexdigest(), "bytes": (root / (sid + ".npy")).stat().st_size,
        "provenance": {"split": "official-test", "repository": repo, "revision": revision, "file": path, "trajectory": 0, "start": start, "range_bytes": raw.downloaded}}
    print(sid, arr.shape, raw.downloaded, flush=True)
    return record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda item: extract(item, root), SOURCES[::2]))
    (root / "index.json").write_text(json.dumps(records, indent=2))
