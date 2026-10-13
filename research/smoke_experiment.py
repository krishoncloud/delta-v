"""Run the real three-arm two-step Euler path on tiny synthetic CPU windows.

Usage: python research/smoke_experiment.py /path/to/domain_fluid.pt
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import experiment
from common import core, sha256


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    checkpoint = Path(sys.argv[1]).resolve()
    torch.set_num_threads(2)
    core.TARGET_HW = (32, 32)  # Same FNO and checkpoint; only shrink the smoke-test pixels.

    with tempfile.TemporaryDirectory(prefix="deltav-euler-smoke-") as scratch:
        root = Path(scratch)
        rng = np.random.default_rng(7)
        records = []
        for trajectory, split in enumerate(("train", "validation", "test")):
            frames = rng.normal(size=(5, 32, 32, 5)).astype(np.float32)
            frames[..., 0] = np.abs(frames[..., 0]) + 1  # Positive density.
            path = root / f"trajectory-{trajectory}.npy"
            np.save(path, frames, allow_pickle=False)
            records.append({"system": core.EULER, "source_revision": "synthetic-smoke",
                            "source_file": "synthetic-euler.hdf5", "trajectory": trajectory,
                            "start_index": 0, "split": split, "path": path.name,
                            "sha256": sha256(path)})
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"checkpoint": {
            "sha256": sha256(checkpoint), "excluded_systems": [core.EULER],
            "exposed_trajectories": [], "provenance_note": "Synthetic smoke test only."
        }, "windows": records}))
        output = root / "results"
        sys.argv = ["experiment.py", "--manifest", str(manifest),
                    "--checkpoint", str(checkpoint), "--output-dir", str(output),
                    "--device", "cpu", "--steps", "2", "--validation-interval", "1",
                    "--trajectory-budgets", "1", "--seeds", "7"]
        experiment.main()
        report = json.loads((output / "results.json").read_text())
        assert {r["method"] for r in report["results"]} == {
            "scratch", "full_finetune", "adapter_only"}
        for result in report["results"]:
            assert result["best_validation_step"] in (1, 2)
            assert np.isfinite(result["best_validation_mse"])
            assert result["metrics"]["per_window"]
            saved = torch.load(output / f"{result['method']}-1-trajectories-seed-7.pt",
                               map_location="cpu", weights_only=True)
            assert saved["model"] and all(torch.is_tensor(v) for v in saved["model"].values())
            model = core.build_fno("cpu")
            if result["method"] == "adapter_only":
                model = experiment.AdaptedFNO(model)
            model.load_state_dict(saved["model"])
            print("PASS", result["method"], "best step", result["best_validation_step"], flush=True)


if __name__ == "__main__":
    main()
