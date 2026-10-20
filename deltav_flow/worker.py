"""Adaptation/inference worker. Talks to the API only through the worker protocol.

    python -m deltav_flow.worker --api http://127.0.0.1:8000 --data-dir work/flow-worker --device cpu
    (token from FLOW_WORKER_TOKEN; source checkpoint from DELTAV_SOURCE_PATH or the pinned HF download)

The worker never trusts the API's validation: it re-validates every input bundle
and recomputes the version hash before fitting. Model weights stay on the
worker's own disk; only metadata, metrics and bounded predictions go back.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import acquisition as Q
from . import adapt as A
from . import bundle as B
from . import engine as E

KINDS = ["fit", "predict", "next", "purge"]


class LeaseLost(RuntimeError):
    pass


class WorkerClient:
    """Worker protocol over any httpx-compatible client (httpx.Client or fastapi TestClient)."""

    def __init__(self, http, token, worker_id, base=""):
        self.http, self.worker_id, self.base = http, worker_id, base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def _post(self, path, payload):
        r = self.http.post(self.base + path, json={"worker_id": self.worker_id, **payload}, headers=self.headers, timeout=60)
        if r.status_code == 409:
            raise LeaseLost(r.text)
        r.raise_for_status()
        return r.json() if r.content else None

    def claim(self, kinds, info):
        r = self.http.post(self.base + "/flow/worker/claim", json={"worker_id": self.worker_id, "kinds": kinds, "info": info},
                           headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json() if r.status_code == 200 else None

    def input(self, job_id, sha):
        r = self.http.get(f"{self.base}/flow/worker/jobs/{job_id}/inputs/{sha}", params={"worker_id": self.worker_id},
                          headers=self.headers, timeout=300)
        if r.status_code == 409:
            raise LeaseLost(r.text)
        r.raise_for_status()
        return r.content

    def heartbeat(self, job_id, progress):
        return self._post(f"/flow/worker/jobs/{job_id}/heartbeat", {"progress": progress})

    def complete(self, job_id, result, runtime):
        return self._post(f"/flow/worker/jobs/{job_id}/complete", {"result": result, "runtime_seconds": runtime})

    def fail(self, job_id, error, runtime, cancelled=False):
        return self._post(f"/flow/worker/jobs/{job_id}/fail", {"error": error, "runtime_seconds": runtime, "cancelled": cancelled})


class Heartbeat(threading.Thread):
    def __init__(self, client, job_id, interval):
        super().__init__(daemon=True)
        self.client, self.job_id, self.interval = client, job_id, interval
        self.progress = {}
        self.cancel = threading.Event()
        self.lost = threading.Event()
        self.stopped = threading.Event()

    def update(self, info):
        self.progress = {**self.progress, **info, "updated_at": time.time()}

    def beat(self):
        try:
            reply = self.client.heartbeat(self.job_id, self.progress)
            if reply and reply.get("cancel_requested"):
                self.cancel.set()
        except LeaseLost:
            self.lost.set()
        except Exception:  # transient network failure; the lease may still be valid
            pass

    def run(self):
        while not self.stopped.wait(self.interval):
            self.beat()

    def should_stop(self):
        return self.cancel.is_set() or self.lost.is_set()


class Worker:
    def __init__(self, client: WorkerClient, data_dir, device="cpu", source_path=None,
                 profiles=("task8-fno-lhs-v1",), heartbeat_seconds=10.0, kinds=KINDS, preadapted_root=None):
        self.client, self.device, self.kinds = client, device, list(kinds)
        self.preadapted_root = Path(preadapted_root) if preadapted_root else None
        self.dir = Path(data_dir)
        (self.dir / "inputs").mkdir(parents=True, exist_ok=True)
        (self.dir / "models").mkdir(exist_ok=True)
        self.source_path, self.profiles, self.heartbeat_seconds = source_path, list(profiles), heartbeat_seconds
        self._source = None
        self._ensemble = (None, None)

    def info(self):
        return {"device": self.device, "profiles": self.profiles, "kinds": self.kinds, "host": socket.gethostname(), "pid": os.getpid()}

    # ----------------------------------------------------------- inputs
    def fetch(self, job, sha):
        path = self.dir / "inputs" / f"{sha}.zip"
        if path.is_file() and E.digest(path) == sha:
            return path.read_bytes()
        data = self.client.input(job["id"], sha)
        if B.sha256_bytes(data) != sha:
            raise ValueError("Downloaded input failed its integrity check")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return data

    def study(self, job):
        version = job["params"]["version"]
        base = B.validate_bundle(self.fetch(job, version["files"]["base"]["sha256"]))
        ranges = [c["range"] for c in base.meta["controls"]]
        additions = [B.validate_addition(self.fetch(job, a["sha256"]), ranges) for a in version["files"]["additions"]]
        study = B.combine(base, additions) if additions else base
        if study.content_sha256 != version["content_sha256"]:
            raise ValueError("Data/model version mismatch: the supplied files do not hash to the requested study version")
        return study

    def source(self, profile, recipe):
        if profile == "machinery-tiny":
            return A.machinery_source_state(recipe["architecture"])
        if self._source is None:
            self._source = E.load_source(E.resolve_source(self.source_path))[0]
        return self._source

    def model_folder(self, model_id, preadapted=None):
        if preadapted:
            if not self.preadapted_root:
                raise FileNotFoundError("This worker has no pre-adapted models.")
            folder = (self.preadapted_root / preadapted / "model").resolve()
            if folder.parent.parent != self.preadapted_root.resolve():
                raise ValueError("Invalid pre-adapted model name")
            return folder
        return self.dir / "models" / model_id

    def ensemble(self, model_id, version_key, preadapted=None):
        folder = self.model_folder(model_id, preadapted)
        if self._ensemble[0] != folder:
            self._ensemble = (None, None)
            if not (folder / "model.json").is_file():
                raise FileNotFoundError("This model's weights are not on the connected worker (deleted, expired or lost). Refit the study version.")
            self._ensemble = (folder, A.Ensemble(folder, self.device))
        model = self._ensemble[1]
        if model.meta["version_key"] != version_key:
            raise ValueError("Data/model version mismatch for this model")
        return model

    # ----------------------------------------------------------- jobs
    def run_once(self):
        job = self.client.claim(self.kinds, self.info())
        if not job:
            return False
        beat = Heartbeat(self.client, job["id"], self.heartbeat_seconds)
        beat.start()
        started = time.perf_counter()
        try:
            result = getattr(self, "do_" + job["kind"])(job, beat)
            if beat.lost.is_set():
                return True
            if beat.cancel.is_set():
                raise E.Cancelled("Cancelled")
            self.client.complete(job["id"], result, time.perf_counter() - started)
        except LeaseLost:
            pass
        except E.Cancelled:
            if not beat.lost.is_set():
                self.client.fail(job["id"], {"code": "cancelled", "message": "Cancelled at your request."},
                                 time.perf_counter() - started, cancelled=True)
        except Exception as exc:
            code = "member_incomplete" if isinstance(exc, E.MemberIncomplete) else (
                "rejected_input" if isinstance(exc, B.BundleRejected) else "worker_error")
            if not beat.lost.is_set():
                self.client.fail(job["id"], {"code": code, "message": str(exc)[:2000],
                                             "detail": traceback.format_exc(limit=3)[-2000:]},
                                 time.perf_counter() - started)
        finally:
            beat.stopped.set()
        return True

    def serve(self, poll=2.0, stop=None):
        stop = stop or threading.Event()
        while not stop.is_set():
            try:
                worked = self.run_once()
            except Exception as exc:  # API unreachable; keep polling
                print(f"[worker] {exc}", flush=True)
                worked = False
            if not worked:
                stop.wait(poll)

    def do_fit(self, job, beat):
        params = job["params"]
        profile = params["profile"]
        if profile not in self.profiles:
            raise ValueError(f"This worker does not run profile {profile!r}")
        beat.update({"stage": "loading", "detail": "verifying inputs"})
        beat.beat()
        study = self.study(job)
        recipe = A.recipe_for(profile)
        beat.update({"stage": "loading", "detail": "loading verified source checkpoint"})
        source = self.source(profile, recipe)
        folder = self.dir / "models" / job["model_id"]
        A.adapt_study(study, job["params"]["version"]["content_sha256"], profile, folder, source, self.device,
                             allow_uncalibrated=params.get("allow_uncalibrated", False),
                             progress=beat.update, should_cancel=beat.should_stop)
        beat.update({"stage": "validating", "detail": "sealing model"})
        return A.model_result(folder)

    def do_predict(self, job, beat):
        p = job["params"]
        model = self.ensemble(job["model_id"], p["version_key"], p.get("preadapted"))
        out = model.predict(np.array(p["points"], float), fields=bool(p.get("fields")))
        result = {"mean": out["mean"].tolist(), "sd": out["sd"].tolist(),
                  "raw_lower": out["raw"][0].tolist(), "raw_upper": out["raw"][1].tolist(),
                  "members": out["members"].tolist()}
        if "calibrated" in out:
            result.update(lower=out["calibrated"][0].tolist(), upper=out["calibrated"][1].tolist())
        if "fields" in out:
            result["fields"] = np.round(out["fields"].astype(np.float64), 6).tolist()
        return result

    def do_next(self, job, beat):
        p = job["params"]
        model = self.ensemble(job["model_id"], p["version_key"], p.get("preadapted"))
        if p.get("candidates"):
            ids, x = p["candidates"]["ids"], np.array(p["candidates"]["x"], float)
            pool_label = "candidates.csv supplied with the study"
        else:
            ids, x = Q.generated_pool(p["ranges"])
            pool_label = f"generated feasible pool ({len(ids)} uniform points in the declared box, seed {Q.POOL_SEED})"
        region_note = None
        if p.get("region"):
            lo, hi = np.array(p["region"], float).T
            keep = np.all((x >= lo - 1e-12) & (x <= hi + 1e-12), axis=1)
            ids, x = [ids[i] for i in np.flatnonzero(keep)], x[keep]
            region_note = f"Limited to your range: {len(ids)} candidate settings inside it."
        sd = None
        if p["policy"] == "uncertainty-diversity-experimental":
            sd = model.predict(x)["sd"]
        chosen, reasons, notes = Q.recommend(p["policy"], ids, x, np.array(p["known_x"], float), p["ranges"],
                                             p["batch"], p.get("acquisition"), sd, p["tolerance"])
        rows = []
        if chosen:
            out = model.predict(x[chosen])
            for k, (i, reason) in enumerate(zip(chosen, reasons)):
                row = {"candidate_id": ids[i], "controls": x[i].tolist(), **reason,
                       "predicted": out["mean"][k].tolist(), "raw_sd": out["sd"][k].tolist()}
                if "calibrated" in out:
                    row["calibrated_interval"] = [out["calibrated"][0][k].tolist(), out["calibrated"][1][k].tolist()]
                rows.append(row)
        if region_note:
            notes.insert(0, region_note)
        return {"rows": rows, "notes": notes, "pool": pool_label}

    def do_purge(self, job, beat):
        removed = []
        for model_id in job["params"].get("model_ids", []):
            folder = (self.dir / "models" / model_id).resolve()
            if folder.parent == (self.dir / "models").resolve() and folder.is_dir():
                shutil.rmtree(folder)
                removed.append(model_id)
            if self._ensemble[0] == folder:
                self._ensemble = (None, None)
        return {"removed": removed}


def main():
    parser = argparse.ArgumentParser(description="Delta-V adaptation worker")
    parser.add_argument("--api", default=os.environ.get("FLOW_API", "http://127.0.0.1:8000"))
    parser.add_argument("--data-dir", default=os.environ.get("FLOW_WORKER_DIR", "work/flow-worker"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), choices=["cpu", "cuda"])
    parser.add_argument("--profiles", default=os.environ.get("FLOW_WORKER_PROFILES", "task8-fno-lhs-v1"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    import httpx

    token = os.environ.get("FLOW_WORKER_TOKEN")
    if not token:
        raise SystemExit("Set FLOW_WORKER_TOKEN (shared secret with the API)")
    client = WorkerClient(httpx.Client(), token, f"{socket.gethostname()}-{os.getpid()}", args.api)
    worker = Worker(client, args.data_dir, args.device, os.environ.get("DELTAV_SOURCE_PATH"),
                    [p.strip() for p in args.profiles.split(",") if p.strip()])
    print(f"[worker] {worker.info()} polling {args.api}", flush=True)
    if args.once:
        worker.run_once()
    else:
        worker.serve()


if __name__ == "__main__":
    main()
