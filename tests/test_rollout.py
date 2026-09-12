import io
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
import main
import deltav_core as core


def test_normalized_recurrence_and_single_conversion(monkeypatch):
    monkeypatch.setattr(core, "TARGET_HW", (8, 8))
    calls = []
    original = core.build_canonical
    def canonical(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(core, "build_canonical", canonical)
    class Increment(torch.nn.Module):
        def forward(self, x):
            return x[:, -4:] + 1
    stats = {"mean": np.array([10, 20, 30, 40]), "std": np.array([2, 3, 4, 5])}
    seeds = [np.ones((12, 16, 5), np.float32) * i for i in range(1, 5)]
    result = core.run_fno_rollout(Increment(), seeds, "euler_multi_quadrants_openBC", stats, 3)
    assert result.shape == (3, 4, 8, 8)
    assert len(calls) == 4  # native/Euler conversion only for the seed
    np.testing.assert_allclose(result[1] - result[0], np.broadcast_to((stats["std"] + 1e-6)[:, None, None], (4, 8, 8)), atol=1e-5)
    np.testing.assert_allclose(result[2] - result[1], result[1] - result[0], atol=1e-5)


def test_rollout_contract_and_rejections(monkeypatch):
    monkeypatch.setitem(core.DATASETS["shear_flow"], "grid", (8, 8))
    monkeypatch.setattr(main, "STATE", {"model": object(), "stats_by": {"shear_flow": {}},
        "samples": {"demo": {"system": "shear_flow"}}, "cache_revision": "r", "checkpoint_epoch": 59})
    monkeypatch.setattr(main, "_load_sample", lambda _: np.ones((5, 8, 8, 4), np.float32))
    called = []
    def infer(*args, **kwargs):
        called.append(1)
        return np.ones((args[4], 4, 256, 256), np.float32)
    monkeypatch.setattr(core, "run_fno_rollout", infer)
    client = TestClient(main.app)
    body = {"system": "shear_flow", "sample_id": "demo", "n_steps": 3, "revision": "r"}
    response = client.post("/predict/rollout", json=body)
    assert response.status_code == 200
    assert np.load(io.BytesIO(response.content)).shape == (3, 4, 256, 256)
    meta = json.loads(response.headers["x-rollout-meta"])
    assert meta["steps"][0]["quality"]["scalar"]["rel_l2"] == 0
    assert meta["steps"][1]["quality"] is None
    for update, code in [({"n_steps": 31}, 422), ({"n_steps": True}, 422), ({"system": "euler"}, 404), ({"revision": "stale"}, 409), ({"sample_id": None, "history_frames": [[1]]}, 422)]:
        assert client.post("/predict/rollout", json={**body, **update}).status_code == code
    assert len(called) == 1
    assert client.post("/events", content=b"x" * 17000).status_code == 413
    main.INFERENCE_SLOT.acquire()
    try:
        busy = client.post("/predict/rollout", json=body)
        assert busy.status_code == 429 and busy.headers["retry-after"] == "10"
    finally:
        main.INFERENCE_SLOT.release()


def test_reject_malicious_npy_header_before_allocation(monkeypatch):
    monkeypatch.setattr(main, "STATE", {"model": object(), "stats_by": {"shear_flow": {}}})
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(payload, {"descr": "<f4", "fortran_order": False, "shape": (4, 999999999, 999999999, 4)})
    response = TestClient(main.app).post("/predict/field", data={"system": "shear_flow"}, files={"file": ("bad.npy", payload.getvalue())})
    assert response.status_code == 422
