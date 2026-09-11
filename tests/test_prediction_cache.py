"""Serving-contract tests with synthetic fields; no checkpoints or raw data downloads."""
import io
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
import main

@pytest.fixture
def client(monkeypatch):
    state = {
        "model": object(), "stats_by": {"shear_flow": {}}, "predictors": {},
        "metrics_by": {}, "checkpoint_epoch": 59, "phase": "ready",
        "samples": {"demo": {"id": "demo", "system": "shear_flow", "dials": {},
                              "t_index": 0, "times": [0,1,2,3,4], "shape": [5,16,16,4]}},
        "prediction_cache": {}, "cache_revision": "test-revision",
    }
    monkeypatch.setattr(main, "STATE", state)
    x = np.linspace(1,2,16*16,dtype=np.float32).reshape(16,16)
    arr = np.stack([np.stack([x+t/20]*4,axis=-1) for t in range(5)])
    monkeypatch.setattr(main, "_load_sample", lambda _: arr)
    def infer(*args, **kwargs):
        truth, _ = main.core.build_canonical(arr[4], "shear_flow")
        return main.core.resample(main.torch.tensor(truth)).numpy() * 1.1
    monkeypatch.setattr(main.core, "run_fno_inference", infer)
    # Deliberately omit the lifespan context so production artifact loading never runs.
    return TestClient(main.app)

def test_quality_exact_threshold_and_zero_reference():
    truth=np.full((4,256,256),20,dtype=np.float32)
    q=main._quality_metrics(truth+3,truth)
    assert q["scalar"]["rel_l2"] == pytest.approx(.15)
    assert q["scalar"]["passes"] is False
    assert q["pressure"]["threshold"] == .15
    assert q["velocity_x"]["threshold"] is None
    assert q["velocity_x"]["passes"] is None
    assert q["scalar"]["ssim"] is None  # constant reference has no data range
    zero=main._quality_metrics(np.zeros_like(truth),np.zeros_like(truth))
    assert zero["scalar"]["rel_l2"] is None
    assert zero["scalar"]["passes"] is None
    assert zero["scalar"]["threshold"] == .15
    assert "undefined" in zero["scalar"]["note"]

def test_ssim_identity():
    x=np.linspace(1,2,256*256,dtype=np.float32).reshape(256,256)
    truth=np.stack([x]*4)
    q=main._quality_metrics(truth,truth)
    assert all(v["ssim"] == pytest.approx(1) for v in q.values())
    assert q["scalar"]["passes"] is True

def test_cache_byte_equivalence_and_etag(client):
    with patch.object(main.core, "run_fno_inference", wraps=main.core.run_fno_inference) as infer:
        first=client.get("/samples/demo/predict")
        second=client.get("/samples/demo/predict")
        assert first.status_code == second.status_code == 200
        assert first.content == second.content
        assert infer.call_count == 1
        assert first.headers["X-Prediction-Cache"] == "MISS"
        assert second.headers["X-Prediction-Cache"] == "HIT"
        assert second.headers["Cache-Control"] == "public, max-age=3600"
        arr=np.load(io.BytesIO(first.content),allow_pickle=False)
        assert arr.shape == (2,4,256,256) and arr.dtype == np.uint8
        import json
        quality=json.loads(first.headers["X-Quality"])
        assert set(quality["scalar"]) == {"rel_l2","rmse","ssim","threshold","passes","note"}
        assert quality["scalar"]["passes"] is True
        assert quality["scalar"]["ssim"] is not None
        conditional=client.get("/samples/demo/predict",headers={"If-None-Match":second.headers["ETag"]})
        assert conditional.status_code == 304 and not conditional.content
        assert infer.call_count == 1
        assert float(second.headers["X-Response-Seconds"]) < .2

def test_simultaneous_cache_misses_compute_once(client):
    with patch.object(main.core, "run_fno_inference", wraps=main.core.run_fno_inference) as infer:
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses=list(pool.map(lambda _:client.get("/samples/demo/predict"),range(3)))
        assert all(r.status_code == 200 for r in responses)
        assert all(r.content == responses[0].content for r in responses)
        assert infer.call_count == 1

def test_catalog_headers_errors_and_revision(client):
    assert client.get("/systems").headers["Cache-Control"] == "public, max-age=300"
    assert client.get("/samples").headers["Cache-Control"] == "public, max-age=300"
    assert client.get("/samples/demo").headers["Cache-Control"] == "public, max-age=86400"
    assert client.get("/samples/unknown/predict").status_code == 404
    assert client.get("/samples/demo/predict?revision=old").status_code == 409
    assert client.get("/samples/demo/predict?revision=test-revision").status_code == 200
    catalog=client.get("/quality")
    import json
    actual=json.loads(client.get("/samples/demo/predict").headers["X-Quality"])
    assert catalog.json()["samples"][0]["quality"] == actual
    main.STATE["model"]=None
    assert client.get("/samples/demo/predict").status_code == 503
    response=client.get("/systems")
    assert response.status_code == 503 and response.headers["Cache-Control"] == "no-store"
    assert client.get("/quality").status_code == 503

def test_startup_precomputes_before_ready(client,monkeypatch,capsys):
    main.STATE["model"]=None
    main.STATE["samples"]={str(i):{"system":"shear_flow"} for i in range(6)}
    model=object()
    monkeypatch.setattr(main, "_pull_artifacts",lambda:None)
    monkeypatch.setattr(main.core,"build_fno",lambda **kwargs:model)
    monkeypatch.setattr(main.core,"load_fno_checkpoint",lambda *a,**kw:(model,59))
    monkeypatch.setattr(main.core,"load_stats",lambda *a:{})
    monkeypatch.setattr(main.core,"load_param_table",lambda *a:{"metrics":[]})
    from types import SimpleNamespace
    monkeypatch.setattr(main.core,"make_predictor",lambda *a:SimpleNamespace(range={}))
    def build(sample_id,m):
        assert m is model and main.STATE["model"] is None
        assert main.health()["model_loaded"] is False
        assert main.STATE["phase"] == "precomputing predictions"
        return {"body":b"", "headers":{}, "quality":{}}
    monkeypatch.setattr(main,"_build_prediction",build)
    main._load_everything()
    assert main.health()["model_loaded"] is True
    assert main.STATE["phase"] == "ready"
    assert len(main.STATE["prediction_cache"]) == 6
    assert capsys.readouterr().out.count("precomputed prediction for") == 6
