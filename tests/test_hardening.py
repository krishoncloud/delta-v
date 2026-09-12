import numpy as np
import pytest
from fastapi.testclient import TestClient
import main
import deltav_core as core


@pytest.fixture(autouse=True)
def _fresh_limiter():
    main.RATE_LIMITER.reset()
    yield
    main.RATE_LIMITER.reset()


def test_openapi_and_docs_are_disabled():
    client = TestClient(main.app)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_security_headers_on_every_response():
    client = TestClient(main.app)
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors" in headers["content-security-policy"]
    assert "script-src 'self'" in headers["content-security-policy"]
    assert headers["referrer-policy"] == "strict-origin-when-cross-origin"


def test_cors_is_not_wildcard():
    client = TestClient(main.app)
    response = client.options("/health", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert response.headers.get("access-control-allow-origin") != "*"
    assert response.headers.get("access-control-allow-origin") != "https://evil.example"


def test_rollout_rejects_caller_supplied_history(monkeypatch):
    monkeypatch.setattr(main, "STATE", {"model": object(), "stats_by": {"shear_flow": {}},
        "samples": {}, "cache_revision": "r", "checkpoint_epoch": 59})
    client = TestClient(main.app)
    body = {"system": "shear_flow", "sample_id": "demo", "n_steps": 1,
            "history_frames": [[[[0.0]]]]}
    assert client.post("/predict/rollout", json=body).status_code == 422
    big = client.post("/predict/rollout", content=b"{" + b"0" * 20000 + b"}",
                      headers={"Content-Type": "application/json"})
    assert big.status_code == 413


def test_per_client_and_global_rate_limits(monkeypatch):
    monkeypatch.setitem(core.DATASETS["shear_flow"], "grid", (8, 8))
    monkeypatch.setattr(main, "STATE", {"model": object(), "stats_by": {"shear_flow": {}},
        "samples": {"demo": {"system": "shear_flow"}}, "cache_revision": "r", "checkpoint_epoch": 59})
    monkeypatch.setattr(main, "_load_sample", lambda _: np.ones((5, 8, 8, 4), np.float32))
    monkeypatch.setattr(core, "run_fno_rollout", lambda *a, **k: np.ones((a[4], 4, 256, 256), np.float32))
    monkeypatch.setitem(main.RATE_LIMITS, "rollout", (2, 60, 3, 60))
    client = TestClient(main.app)
    body = {"system": "shear_flow", "sample_id": "demo", "n_steps": 1}

    a = {"X-Forwarded-For": "1.1.1.1"}
    assert client.post("/predict/rollout", json=body, headers=a).status_code == 200
    assert client.post("/predict/rollout", json=body, headers=a).status_code == 200
    third = client.post("/predict/rollout", json=body, headers=a)
    assert third.status_code == 429 and int(third.headers["retry-after"]) >= 1

    b = {"X-Forwarded-For": "2.2.2.2"}
    assert client.post("/predict/rollout", json=body, headers=b).status_code == 200
    assert client.post("/predict/rollout", json=body, headers=b).status_code == 429  # global cap

    spoof = {"X-Forwarded-For": "9.9.9.9, 1.1.1.1"}  # rightmost hop is what counts
    assert client.post("/predict/rollout", json=body, headers=spoof).status_code == 429


def test_rate_limiter_bounds_its_memory():
    limiter = main.RateLimiter()
    limiter.MAX_KEYS = 50
    for i in range(200):
        limiter.hit(f"k{i}", 10, 60)
    assert len(limiter._hits) <= 50
