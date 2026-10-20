"""Durable private store: one SQLite ledger plus content-addressed immutable files.

Single-instance prototype. SQLite (WAL) is safe for one API process with threads;
it is NOT shared with a worker on another host. Remote workers reach state only
through the authenticated worker protocol in api.py. Run one API instance per
data directory; horizontal scaling needs a real database/queue (see docs/flow/WORKER.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS owners(id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS studies(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, name TEXT NOT NULL, data_kind TEXT NOT NULL,
    example_id TEXT, created_at REAL NOT NULL, deleted_at REAL);
CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY, study_id TEXT NOT NULL, number INTEGER NOT NULL, parent_id TEXT,
    created_at REAL NOT NULL, content_sha256 TEXT NOT NULL, files TEXT NOT NULL, report TEXT NOT NULL, origin TEXT NOT NULL,
    UNIQUE(study_id, number));
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, kind TEXT NOT NULL, study_id TEXT, version_id TEXT,
    model_id TEXT, params TEXT NOT NULL, dedupe_key TEXT, status TEXT NOT NULL, stage TEXT, progress TEXT, error TEXT,
    result TEXT, attempts INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_expires REAL, cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, started_at REAL, finished_at REAL, runtime_seconds REAL NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, study_id TEXT NOT NULL, version_id TEXT NOT NULL,
    job_id TEXT NOT NULL, profile TEXT NOT NULL, worker_id TEXT, meta TEXT NOT NULL, status TEXT NOT NULL,
    created_at REAL NOT NULL, expires_at REAL);
CREATE TABLE IF NOT EXISTS recommendations(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, study_id TEXT NOT NULL, version_id TEXT NOT NULL,
    model_id TEXT, job_id TEXT NOT NULL, created_at REAL NOT NULL, rows TEXT NOT NULL, meta TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ledger(id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, study_id TEXT NOT NULL, kind TEXT NOT NULL,
    amount REAL NOT NULL, unit TEXT NOT NULL, job_id TEXT, note TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS scenarios(study_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, inputs TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS workers(id TEXT PRIMARY KEY, last_seen REAL NOT NULL, info TEXT NOT NULL);
"""

ACTIVE = ("queued", "running")
TERMINAL = ("ready", "done", "failed", "cancelled")


def now():
    return time.time()


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class NotFound(LookupError):
    pass


class Conflict(RuntimeError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "ledger.sqlite3", check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")

    def one(self, sql, *args):
        with self._lock:
            row = self.db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def all(self, sql, *args):
        with self._lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # ---------------------------------------------------------------- blobs
    def put_blob(self, data: bytes) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        path = self.root / "blobs" / sha[:2] / (sha + ".zip")
        if not path.is_file():
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(".tmp" + secrets.token_hex(4))
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return {"sha256": sha, "bytes": len(data)}

    def blob_path(self, sha: str) -> Path:
        if not (len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            raise NotFound("blob")
        path = self.root / "blobs" / sha[:2] / (sha + ".zip")
        if not path.is_file():
            raise NotFound("blob")
        return path

    def read_blob(self, sha: str) -> bytes:
        data = self.blob_path(sha).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"Stored file {sha[:12]} failed its integrity check")
        return data

    # ---------------------------------------------------------------- owners
    def create_owner(self):
        token = secrets.token_urlsafe(32)
        owner = new_id("own")
        with self.tx() as db:
            db.execute("INSERT INTO owners VALUES (?,?,?)", (owner, token_hash(token), now()))
        return owner, token

    def owner_for(self, token: str | None):
        if not token or len(token) > 200:
            return None
        # Lookup is by SHA-256 of a 256-bit random token, so timing reveals nothing usable.
        row = self.one("SELECT id FROM owners WHERE token_hash=?", token_hash(token))
        return row["id"] if row else None

    # ---------------------------------------------------------------- studies / versions
    def create_study(self, owner, name, data_kind, blob, content_sha256, report, example_id=None):
        study, version = new_id("std"), new_id("ver")
        with self.tx() as db:
            db.execute("INSERT INTO studies VALUES (?,?,?,?,?,?,NULL)", (study, owner, name, data_kind, example_id, now()))
            db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?)",
                       (version, study, 1, None, now(), content_sha256, json.dumps({"base": blob, "additions": []}),
                        json.dumps(report), "example" if example_id else "upload"))
        return study, version

    def study(self, owner, study_id):
        row = self.one("SELECT * FROM studies WHERE id=? AND owner_id=? AND deleted_at IS NULL", study_id, owner)
        if not row:
            raise NotFound("study")
        return row

    def studies(self, owner):
        return self.all("SELECT * FROM studies WHERE owner_id=? AND deleted_at IS NULL ORDER BY created_at DESC", owner)

    def versions(self, study_id):
        rows = self.all("SELECT * FROM versions WHERE study_id=? ORDER BY number", study_id)
        for r in rows:
            r["files"] = json.loads(r["files"])
            r["report"] = json.loads(r["report"])
        return rows

    def version(self, owner, version_id):
        row = self.one("SELECT v.* FROM versions v JOIN studies s ON s.id=v.study_id "
                       "WHERE v.id=? AND s.owner_id=? AND s.deleted_at IS NULL", version_id, owner)
        if not row:
            raise NotFound("version")
        row["files"] = json.loads(row["files"])
        row["report"] = json.loads(row["report"])
        return row

    def add_version(self, owner, study_id, parent_id, blob, content_sha256, report):
        with self.tx() as db:
            latest = db.execute("SELECT id, number, files FROM versions WHERE study_id=? ORDER BY number DESC LIMIT 1", (study_id,)).fetchone()
            if latest["id"] != parent_id:
                raise Conflict("The study has a newer version; reload and add runs to the latest version.")
            files = json.loads(latest["files"])
            files["additions"] = files["additions"] + [blob]
            version = new_id("ver")
            db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?)",
                       (version, study_id, latest["number"] + 1, parent_id, now(), content_sha256, json.dumps(files),
                        json.dumps(report), "addition"))
        return version

    def delete_study(self, owner, study_id):
        self.study(owner, study_id)
        with self.tx() as db:
            db.execute("UPDATE studies SET deleted_at=? WHERE id=?", (now(), study_id))
            db.execute("UPDATE jobs SET cancel_requested=1 WHERE study_id=? AND status='running'", (study_id,))
            db.execute("UPDATE jobs SET status='cancelled', finished_at=?, updated_at=? WHERE study_id=? AND status='queued'",
                       (now(), now(), study_id))
            models = [r["id"] for r in db.execute("SELECT id FROM models WHERE study_id=? AND status='ready'", (study_id,))]
            db.execute("UPDATE models SET status='deleted' WHERE study_id=?", (study_id,))
        def shas(files):
            return {files["base"]["sha256"], *(a["sha256"] for a in files["additions"])}

        mine = set().union(*(shas(v["files"]) for v in self.versions(study_id)))
        live = set().union(set(), *(shas(json.loads(r["files"])) for r in self.all(
            "SELECT v.files FROM versions v JOIN studies s ON s.id=v.study_id WHERE s.deleted_at IS NULL")))
        for sha in mine - live:
            try:
                self.blob_path(sha).unlink()
            except NotFound:
                pass
        return models

    # ---------------------------------------------------------------- jobs
    def submit(self, owner, kind, params, study_id=None, version_id=None, model_id=None, dedupe_key=None,
               max_active_owner=1, max_queued=20):
        """Create a job, or return the existing live job with the same dedupe key (idempotent)."""
        with self.tx() as db:
            if dedupe_key:
                row = db.execute("SELECT * FROM jobs WHERE owner_id=? AND dedupe_key=? AND status IN ('queued','running','ready','done') "
                                 "ORDER BY created_at DESC LIMIT 1", (owner, dedupe_key)).fetchone()
                if row:
                    return _decode_job(row), False
            active = db.execute("SELECT COUNT(*) FROM jobs WHERE owner_id=? AND kind=? AND status IN ('queued','running')",
                                (owner, kind)).fetchone()[0]
            if active >= max_active_owner:
                raise Conflict(f"You already have {active} active {kind} job(s); wait for it to finish or cancel it.")
            queued = db.execute("SELECT COUNT(*) FROM jobs WHERE kind=? AND status='queued'", (kind,)).fetchone()[0]
            if queued >= max_queued:
                raise OverflowError("The adaptation queue is full; try again later.")
            job = new_id("job")
            t = now()
            db.execute("INSERT INTO jobs (id, owner_id, kind, study_id, version_id, model_id, params, dedupe_key, status, stage, "
                       "progress, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?, 'queued', 'queued', '{}', ?, ?)",
                       (job, owner, kind, study_id, version_id, model_id, json.dumps(params), dedupe_key, t, t))
        return self.job(owner, job), True

    def job(self, owner, job_id):
        row = self.one("SELECT * FROM jobs WHERE id=? AND owner_id=?", job_id, owner)
        if not row:
            raise NotFound("job")
        return _decode_job(row)

    def jobs_for_study(self, owner, study_id):
        return [_decode_job(r) for r in self.all("SELECT * FROM jobs WHERE owner_id=? AND study_id=? ORDER BY created_at", owner, study_id)]

    def request_cancel(self, owner, job_id):
        job = self.job(owner, job_id)
        with self.tx() as db:
            if job["status"] == "queued":
                db.execute("UPDATE jobs SET status='cancelled', stage='cancelled', finished_at=?, updated_at=? WHERE id=?",
                           (now(), now(), job_id))
            elif job["status"] == "running":
                db.execute("UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?", (now(), job_id))
        return self.job(owner, job_id)

    def claim(self, worker_id, kinds, lease_seconds):
        """Lease the oldest queued job, or one whose previous worker's lease expired."""
        t = now()
        with self.tx() as db:
            marks = ",".join("?" * len(kinds))
            row = db.execute(f"SELECT * FROM jobs WHERE kind IN ({marks}) AND (status='queued' OR (status='running' AND lease_expires < ?)) "
                             "AND cancel_requested=0 ORDER BY created_at LIMIT 1", (*kinds, t)).fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET status='running', stage='loading', lease_owner=?, lease_expires=?, attempts=attempts+1, "
                       "started_at=COALESCE(started_at, ?), updated_at=? WHERE id=?",
                       (worker_id, t + lease_seconds, t, t, row["id"]))
        return _decode_job(self.one("SELECT * FROM jobs WHERE id=?", row["id"]))

    def leased(self, job_id, worker_id):
        row = self.one("SELECT * FROM jobs WHERE id=?", job_id)
        if not row or row["status"] != "running" or row["lease_owner"] != worker_id:
            raise Conflict("This worker does not hold the job lease")
        return _decode_job(row)

    def heartbeat(self, job_id, worker_id, progress, lease_seconds):
        self.leased(job_id, worker_id)
        t = now()
        stage = progress.get("stage") if isinstance(progress, dict) else None
        with self.tx() as db:
            db.execute("UPDATE jobs SET lease_expires=?, progress=?, stage=COALESCE(?, stage), updated_at=? WHERE id=?",
                       (t + lease_seconds, json.dumps(progress), stage, t, job_id))
        return self.one("SELECT cancel_requested, lease_expires FROM jobs WHERE id=?", job_id)

    def finish(self, job_id, worker_id, status, result=None, error=None, runtime_seconds=0.0):
        job = self.leased(job_id, worker_id)
        t = now()
        with self.tx() as db:
            db.execute("UPDATE jobs SET status=?, stage=?, result=?, error=?, finished_at=?, updated_at=?, lease_owner=NULL, "
                       "runtime_seconds=runtime_seconds+? WHERE id=?",
                       (status, status, json.dumps(result) if result is not None else None,
                        json.dumps(error) if error is not None else None, t, t, float(runtime_seconds), job_id))
            if runtime_seconds:
                db.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?)",
                           (new_id("led"), job["owner_id"], job["study_id"] or "", "worker_" + job["kind"], float(runtime_seconds),
                            "seconds", job_id, f"measured worker wall time on {worker_id}", t))
        return job

    # ---------------------------------------------------------------- models
    def create_model(self, job, worker_id, meta, retention_seconds):
        t = now()
        with self.tx() as db:
            db.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (job["model_id"], job["owner_id"], job["study_id"], job["version_id"], job["id"], job["params"]["profile"],
                        worker_id, json.dumps(meta), "ready", t, t + retention_seconds if retention_seconds else None))

    def attach_model(self, owner, study_id, version_id, model_id, profile, worker_id, meta):
        """Register an already-sealed model (the worked example's pre-adapted model) for one owner's study."""
        with self.tx() as db:
            db.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (model_id, owner, study_id, version_id, "preadapted", profile, worker_id, json.dumps(meta), "ready", now(), None))

    def model(self, owner, model_id):
        row = self.one("SELECT m.* FROM models m JOIN studies s ON s.id=m.study_id WHERE m.id=? AND m.owner_id=? "
                       "AND s.deleted_at IS NULL", model_id, owner)
        if not row:
            raise NotFound("model")
        row["meta"] = json.loads(row["meta"])
        return row

    def models_for_study(self, owner, study_id):
        rows = self.all("SELECT * FROM models WHERE owner_id=? AND study_id=? ORDER BY created_at", owner, study_id)
        for r in rows:
            r["meta"] = json.loads(r["meta"])
        return rows

    # ---------------------------------------------------------------- recommendations, ledger, scenarios, workers
    def save_recommendation(self, job, rows, meta):
        rec = new_id("rec")
        with self.tx() as db:
            db.execute("INSERT INTO recommendations VALUES (?,?,?,?,?,?,?,?,?)",
                       (rec, job["owner_id"], job["study_id"], job["version_id"], job["model_id"], job["id"], now(),
                        json.dumps(rows), json.dumps(meta)))
        return rec

    def recommendations(self, owner, study_id):
        rows = self.all("SELECT * FROM recommendations WHERE owner_id=? AND study_id=? ORDER BY created_at", owner, study_id)
        for r in rows:
            r["rows"] = json.loads(r["rows"])
            r["meta"] = json.loads(r["meta"])
        return rows

    def ledger(self, owner, study_id):
        return self.all("SELECT * FROM ledger WHERE owner_id=? AND study_id=? ORDER BY created_at", owner, study_id)

    def add_charge(self, owner, study_id, amount, currency, note):
        with self.tx() as db:
            db.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?)",
                       (new_id("led"), owner, study_id, "charge", float(amount), currency, None, note, now()))

    def scenario(self, owner, study_id):
        row = self.one("SELECT inputs FROM scenarios WHERE study_id=? AND owner_id=?", study_id, owner)
        return json.loads(row["inputs"]) if row else None

    def set_scenario(self, owner, study_id, inputs):
        with self.tx() as db:
            db.execute("INSERT INTO scenarios VALUES (?,?,?,?) ON CONFLICT(study_id) DO UPDATE SET inputs=excluded.inputs, "
                       "updated_at=excluded.updated_at", (study_id, owner, json.dumps(inputs), now()))

    def seen_worker(self, worker_id, info):
        with self.tx() as db:
            db.execute("INSERT INTO workers VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen, "
                       "info=excluded.info", (worker_id, now(), json.dumps(info)))

    def workers(self, within_seconds):
        rows = self.all("SELECT * FROM workers WHERE last_seen > ? ORDER BY last_seen DESC", now() - within_seconds)
        for r in rows:
            r["info"] = json.loads(r["info"])
        return rows

    def close(self):
        with self._lock:
            self.db.close()

    def wipe(self):
        """Test helper: remove every stored file and row."""
        self.close()
        shutil.rmtree(self.root, ignore_errors=True)


def _decode_job(row):
    row = dict(row)
    for key in ("params", "progress", "error", "result"):
        row[key] = json.loads(row[key]) if row.get(key) else ({} if key in ("params", "progress") else None)
    row["cancel_requested"] = bool(row["cancel_requested"])
    return row
