import importlib
import threading
import time
import uuid

from fastapi.testclient import TestClient


def test_independent_outputs_idempotency_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("HIRES_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HIRES_WORK_DIR", str(tmp_path / "work"))
    import app as module
    module = importlib.reload(module)

    def map_build(job, work, cache, target, progress, cancelled):
        progress("packaging", 1, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"mock-map")
        return 8, "f" * 64

    def gkot_build(job, work, cache, target, progress, cancelled):
        raise RuntimeError("point source failed")

    monkeypatch.setattr(module, "build_map", map_build)
    monkeypatch.setattr(module, "build_gkot", gkot_build)
    request = {"requestId": str(uuid.uuid4()),
               "center": {"lat": 46.378333, "lon": 13.836667},
               "widthMeters": 1000, "heightMeters": 1000, "includeGkot": True}
    with TestClient(module.app) as client:
        created = client.post("/jobs", json=request)
        assert created.status_code == 202
        identifier = created.json()["id"]
        assert client.post("/jobs", json=request).json()["id"] == identifier
        changed = dict(request, widthMeters=1100)
        assert client.post("/jobs", json=changed).status_code == 409
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.get(f"/jobs/{identifier}").json()["artifacts"]
            if status["map"]["state"] == "ready" and status["gkot"]["state"] == "failed":
                break
            time.sleep(0.05)
        assert status["map"]["state"] == "ready"
        assert status["gkot"]["state"] == "failed"
        assert client.get(f"/jobs/{identifier}/files/map").content == b"mock-map"
        assert client.get(f"/jobs/{identifier}/files/gkot").status_code == 409
        assert client.delete(f"/jobs/{identifier}").json()["deleted"] is True
        assert client.get(f"/jobs/{identifier}").status_code == 404
        assert not (tmp_path / "data" / "packages" / f"{identifier}-map.zip").exists()


def test_triglav_rectangle_crosses_four_tiles():
    from sources import bounds_for, cells_for
    bounds = bounds_for({"lat": 46.378333, "lon": 13.836667}, 1000, 1000)
    assert cells_for(bounds) == [(410, 137), (410, 138), (411, 137), (411, 138)]


def test_complete_partial_download_is_promoted_after_range_416(tmp_path, monkeypatch):
    import sources

    destination = tmp_path / "tile.laz"
    destination.with_suffix(".laz.part").write_bytes(b"complete")

    class Response:
        status_code = 416
        headers = {"Content-Range": "bytes */8"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(sources.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sources, "_validate", lambda *args: None)
    path = sources._download_locked({"cell": "410_138"}, "gkot", "https://assets.flycom.si/x",
                                    destination, None, None)
    assert path.read_bytes() == b"complete"
    assert not destination.with_suffix(".laz.part").exists()


def test_delete_stops_active_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("HIRES_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HIRES_WORK_DIR", str(tmp_path / "work"))
    import app as module
    module = importlib.reload(module)
    started = threading.Event()

    def waiting_build(job, work, cache, target, progress, cancelled):
        started.set()
        while not cancelled():
            time.sleep(0.01)
        raise module.SourceError("CANCELLED", "Job was deleted")

    monkeypatch.setattr(module, "build_map", waiting_build)
    request = {"requestId": str(uuid.uuid4()),
               "center": {"lat": 46.378333, "lon": 13.836667},
               "widthMeters": 1000, "heightMeters": 1000}
    with TestClient(module.app) as client:
        identifier = client.post("/jobs", json=request).json()["id"]
        assert started.wait(3)
        assert client.delete(f"/jobs/{identifier}").json()["deleted"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with module.connect() as con:
                state = con.execute("SELECT state FROM artifacts WHERE job_id=?", (identifier,)).fetchone()[0]
            if state == "deleted":
                break
            time.sleep(0.02)
        assert state == "deleted"
        assert client.get(f"/jobs/{identifier}").status_code == 404
