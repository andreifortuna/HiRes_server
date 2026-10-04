"""Small loopback-only build API for Slovenian orthophoto, DMR, and GKOT."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from build import build_gkot, build_map
from sources import SourceError, bounds_for

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("HIRES_DATA_DIR", ROOT / "data")).resolve()
WORK = Path(os.environ.get("HIRES_WORK_DIR", Path("C:/tmp/hires-server/work") if os.name == "nt" else Path("/tmp/hires-server/work"))).resolve()
DB = DATA / "jobs.sqlite3"
STOP = threading.Event()
LOG = logging.getLogger("hires_server")


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    requestId: uuid.UUID
    center: dict[str, float]
    widthMeters: float = Field(ge=100)
    heightMeters: float = Field(ge=100)
    includeGkot: bool = False

    @model_validator(mode="after")
    def check(self):
        if (set(self.center) != {"lat", "lon"}
                or not all(math.isfinite(v) for v in self.center.values())
                or not -90 <= self.center["lat"] <= 90
                or not -180 <= self.center["lon"] <= 180):
            raise ValueError("center must contain finite WGS84 lat and lon")
        return self


def connect():
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


def initialize():
    for directory in (DATA, DATA / "packages", DATA / "cache", WORK):
        directory.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
            request_json TEXT NOT NULL, bounds_json TEXT NOT NULL,
            created_at REAL NOT NULL, deleted INTEGER NOT NULL DEFAULT 0)""")
        con.execute("""CREATE TABLE IF NOT EXISTS artifacts (
            job_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
            stage TEXT, completed INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL DEFAULT 0, size_bytes INTEGER,
            sha256 TEXT, error_code TEXT, error_message TEXT,
            updated_at REAL NOT NULL,
            PRIMARY KEY(job_id, kind), FOREIGN KEY(job_id) REFERENCES jobs(id))""")
        con.execute("UPDATE artifacts SET state='queued',stage=NULL,completed=0,total=0 "
                    "WHERE state='running' AND job_id IN (SELECT id FROM jobs WHERE deleted=0)")


def _row(job_id: str):
    with connect() as con:
        return con.execute("SELECT * FROM jobs WHERE id=? AND deleted=0", (job_id,)).fetchone()


def _deleted(job_id: str) -> bool:
    with connect() as con:
        row = con.execute("SELECT deleted FROM jobs WHERE id=?", (job_id,)).fetchone()
        return not row or bool(row[0])


def _response(job_id: str):
    with connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=? AND deleted=0", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found")
        artifacts = con.execute("SELECT * FROM artifacts WHERE job_id=? ORDER BY kind", (job_id,)).fetchall()
    return {
        "id": row["id"], "requestId": row["request_id"],
        "center": json.loads(row["request_json"])["center"],
        "widthMeters": json.loads(row["request_json"])["widthMeters"],
        "heightMeters": json.loads(row["request_json"])["heightMeters"],
        "crs": "EPSG:3794", "boundsMetres": json.loads(row["bounds_json"]),
        "createdAt": row["created_at"],
        "artifacts": {a["kind"]: {
            "state": a["state"], "stage": a["stage"],
            "completed": a["completed"], "total": a["total"],
            "sizeBytes": a["size_bytes"], "sha256": a["sha256"],
            "downloadUrl": f"/jobs/{job_id}/files/{a['kind']}" if a["state"] == "ready" else None,
            "error": {"code": a["error_code"], "message": a["error_message"]}
                     if a["error_code"] else None,
        } for a in artifacts},
    }


def _claim(kind: str):
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT a.job_id,j.request_json,j.bounds_json FROM artifacts a "
                          "JOIN jobs j ON j.id=a.job_id WHERE a.kind=? AND a.state='queued' "
                          "AND j.deleted=0 ORDER BY j.created_at,j.id LIMIT 1", (kind,)).fetchone()
        if row:
            con.execute("UPDATE artifacts SET state='running',stage='starting',updated_at=? "
                        "WHERE job_id=? AND kind=?", (time.time(), row["job_id"], kind))
    if not row:
        return None
    job = json.loads(row["request_json"])
    job["id"] = row["job_id"]
    job["bounds"] = json.loads(row["bounds_json"])
    return job


def _progress(job_id: str, kind: str, stage: str, completed: int, total: int):
    with connect() as con:
        con.execute("UPDATE artifacts SET stage=?,completed=?,total=?,updated_at=? "
                    "WHERE job_id=? AND kind=? AND state='running'",
                    (stage, completed, total, time.time(), job_id, kind))


def _safe_rmtree(root: Path, child: Path):
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    if resolved_child.parent != resolved_root or len(child.name) != 32 or any(c not in "0123456789abcdef" for c in child.name):
        raise RuntimeError("Unsafe work directory cleanup path")
    if resolved_child.exists():
        shutil.rmtree(resolved_child)


def _safe_remove_artifact(job_work: Path, artifact_work: Path):
    if (job_work.resolve().parent != WORK.resolve()
            or artifact_work.resolve().parent != job_work.resolve()
            or artifact_work.name not in {"map", "gkot"}):
        raise RuntimeError("Unsafe artifact cleanup path")
    if artifact_work.exists():
        shutil.rmtree(artifact_work)


def _worker(kind: str):
    while not STOP.is_set():
        job_id = None
        try:
            job = _claim(kind)
            if not job:
                STOP.wait(0.5)
                continue
            job_id = job["id"]
            work = WORK / job_id / kind
            target = DATA / "packages" / f"{job_id}-{kind}.zip"
            # A restarted job discards incomplete derived files, but cached source
            # downloads keep their resumable .part files.
            job_work = WORK / job_id
            if job_work.exists() and not job_work.is_dir():
                raise RuntimeError("Invalid work directory")
            if work.exists():
                _safe_remove_artifact(job_work, work)
            work.mkdir(parents=True)
            target.unlink(missing_ok=True)
            cancelled = lambda: _deleted(job_id)
            progress = lambda stage, done, total: _progress(job_id, kind, stage, done, total)
            try:
                build = build_map if kind == "map" else build_gkot
                size, digest = build(job, work, DATA / "cache", target, progress, cancelled)
                if cancelled():
                    target.unlink(missing_ok=True)
                    with connect() as con:
                        con.execute("UPDATE artifacts SET state='deleted',stage=NULL,updated_at=? "
                                    "WHERE job_id=? AND kind=? AND state='running'",
                                    (time.time(), job_id, kind))
                else:
                    with connect() as con:
                        con.execute("UPDATE artifacts SET state='ready',stage=NULL,completed=0,total=0,"
                                    "size_bytes=?,sha256=?,updated_at=? WHERE job_id=? AND kind=? AND state='running'",
                                    (size, digest, time.time(), job_id, kind))
            except Exception as error:
                target.unlink(missing_ok=True)
                if cancelled():
                    with connect() as con:
                        con.execute("UPDATE artifacts SET state='deleted',stage=NULL,updated_at=? "
                                    "WHERE job_id=? AND kind=? AND state='running'",
                                    (time.time(), job_id, kind))
                else:
                    code = error.code if isinstance(error, SourceError) else "BUILD_FAILED"
                    LOG.exception("%s %s failed", job_id, kind)
                    with connect() as con:
                        con.execute("UPDATE artifacts SET state='failed',error_code=?,error_message=?,updated_at=? "
                                    "WHERE job_id=? AND kind=? AND state='running'",
                                    (code, str(error)[:1000], time.time(), job_id, kind))
            finally:
                if work.exists():
                    _safe_remove_artifact(job_work, work)
                if job_work.exists() and not any(job_work.iterdir()):
                    _safe_rmtree(WORK, job_work)
        except Exception:
            LOG.exception("Worker %s encountered an unexpected error", kind)
            if job_id and not _deleted(job_id):
                with connect() as con:
                    con.execute("UPDATE artifacts SET state='failed',error_code='BUILD_FAILED',"
                                "error_message='Worker failed while preparing or cleaning the job',updated_at=? "
                                "WHERE job_id=? AND kind=? AND state='running'",
                                (time.time(), job_id, kind))
            STOP.wait(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STOP.clear()
    initialize()
    logging.basicConfig(filename=DATA / "server.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    threads = [threading.Thread(target=_worker, args=(kind,), name=f"{kind}-worker", daemon=True)
               for kind in ("map", "gkot")]
    for thread in threads:
        thread.start()
    app.state.threads = threads
    yield
    STOP.set()
    for thread in threads:
        thread.join(timeout=2)


app = FastAPI(title="HiRes Slovenia package server", lifespan=lifespan)


@app.get("/health")
def health():
    live = all(thread.is_alive() for thread in app.state.threads)
    if not live:
        raise HTTPException(503, "Worker stopped")
    return {"status": "ok"}


@app.post("/jobs", status_code=202)
def create_job(request: BuildRequest):
    data = request.model_dump(mode="json")
    request_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    bounds = bounds_for(data["center"], data["widthMeters"], data["heightMeters"])
    if not all(math.isfinite(v) for v in bounds.values()):
        raise HTTPException(422, "Requested bounds are invalid")
    # This is a processing-safety bound, not a user quota.
    if math.ceil(data["widthMeters"] / 0.5) * math.ceil(data["heightMeters"] / 0.5) > 100_000_000:
        raise HTTPException(413, "Requested height grid exceeds this server's memory capacity")
    identifier = uuid.uuid4().hex
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute("SELECT id,request_json,deleted FROM jobs WHERE request_id=?",
                               (data["requestId"],)).fetchone()
        if existing:
            if existing["request_json"] != request_json or existing["deleted"]:
                raise HTTPException(409, "requestId has already been used for a different or deleted job")
            identifier = existing["id"]
        else:
            con.execute("INSERT INTO jobs(id,request_id,request_json,bounds_json,created_at) VALUES(?,?,?,?,?)",
                        (identifier, data["requestId"], request_json, json.dumps(bounds), time.time()))
            for kind in (("map", "gkot") if data["includeGkot"] else ("map",)):
                con.execute("INSERT INTO artifacts(job_id,kind,state,updated_at) VALUES(?,?,'queued',?)",
                            (identifier, kind, time.time()))
    return _response(identifier)


@app.get("/jobs")
def list_jobs():
    with connect() as con:
        rows = con.execute("SELECT id FROM jobs WHERE deleted=0 ORDER BY created_at DESC").fetchall()
    return {"jobs": [_response(row["id"]) for row in rows]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _response(job_id)


@app.api_route("/jobs/{job_id}/files/{kind}", methods=["GET", "HEAD"])
def get_file(job_id: str, kind: str):
    if kind not in ("map", "gkot"):
        raise HTTPException(404, "Artifact not found")
    with connect() as con:
        row = con.execute("SELECT a.* FROM artifacts a JOIN jobs j ON j.id=a.job_id "
                          "WHERE j.id=? AND j.deleted=0 AND a.kind=?", (job_id, kind)).fetchone()
    if not row:
        raise HTTPException(404, "Artifact not found")
    if row["state"] != "ready":
        raise HTTPException(409, f"Artifact is {row['state']}")
    path = DATA / "packages" / f"{job_id}-{kind}.zip"
    if not path.is_file():
        raise HTTPException(503, "Completed artifact file is missing")
    return FileResponse(path, media_type="application/zip", filename=f"hires-{job_id}-{kind}.zip",
                        headers={"ETag": '"' + row["sha256"] + '"',
                                 "X-Checksum-SHA256": row["sha256"]})


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT deleted FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return {"deleted": True}
        running = con.execute("SELECT COUNT(*) FROM artifacts WHERE job_id=? AND state='running'", (job_id,)).fetchone()[0]
        con.execute("UPDATE jobs SET deleted=1 WHERE id=?", (job_id,))
        con.execute("UPDATE artifacts SET state='deleted',stage=NULL WHERE job_id=? AND state!='running'", (job_id,))
    cleanup_pending = bool(running)
    for kind in ("map", "gkot"):
        try:
            (DATA / "packages" / f"{job_id}-{kind}.zip").unlink(missing_ok=True)
        except OSError:
            cleanup_pending = True
            LOG.warning("Package deletion deferred for %s %s", job_id, kind)
    return {"deleted": True, "cleanupPending": cleanup_pending}
