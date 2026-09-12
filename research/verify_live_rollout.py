"""Sequential live smoke tests; no model or source-data downloads."""
import argparse
import io
import json
import time
from pathlib import Path
import httpx
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = "https://krishmalik-delta-v.hf.space"
    report = []
    with httpx.Client(timeout=180) as client:
        health = client.get(base + "/health").json()
        assert health["model_loaded"], health
        samples = client.get(base + "/samples").json()
        clips = [s for s in samples if s["id"].startswith("rollout-test-")]
        assert len(clips) == 3, "Extended clips not yet deployed"
        for sample in clips:
            started = time.perf_counter()
            response = client.post(base + "/predict/rollout", json={"system": sample["system"], "sample_id": sample["id"], "n_steps": 7, "revision": health["cache_revision"]})
            response.raise_for_status()
            predicted = np.load(io.BytesIO(response.content), allow_pickle=False)
            meta = json.loads(response.headers["x-rollout-meta"])
            assert predicted.shape == (7, 4, 256, 256) and np.isfinite(predicted).all()
            assert all(s["quality"] is not None for s in meta["steps"])
            # Verify step 1 matches the existing, separate single-step API.
            first = client.post(base + "/predict/field", data={"system": sample["system"], "sample_id": sample["id"]})
            first.raise_for_status()
            np.testing.assert_allclose(predicted[0], np.load(io.BytesIO(first.content)), atol=1e-6, rtol=1e-6)
            record = {"sample": sample["id"], "shape": list(predicted.shape), "wall_seconds_including_single_step_check": time.perf_counter()-started,
                "metadata": meta, "first_step_matches_single_step": True}
            report.append(record)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(sample["id"], "inference_seconds", meta["inference_seconds"], "scalar_L2", [round(s["quality"]["scalar"]["rel_l2"], 4) for s in meta["steps"]], flush=True)
