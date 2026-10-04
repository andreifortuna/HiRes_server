"""Add Cocos runtime terrain assets to an already completed map ZIP."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import zipfile

import lerc
import numpy as np

from render_terrain import write_render_assets

ROOT = Path(__file__).resolve().parent


def upgrade(job_id: str) -> None:
    if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
        raise ValueError("Invalid job ID")
    data = ROOT / "data"
    path = data / "packages" / f"{job_id}-map.zip"
    if not path.is_file():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory(prefix="hires-render-", dir="C:/tmp") as directory:
        work = Path(directory)
        with zipfile.ZipFile(path) as source:
            manifest = json.loads(source.read("manifest.json"))
            terrain = json.loads(source.read("terrain.json"))
            status, heights, _ = lerc.decode(source.read("terrain-height.lerc"))
            if status != 0 or heights.ndim != 2 or not np.isfinite(heights).all():
                raise ValueError("Invalid LERC height grid")
            render = write_render_assets(heights, terrain["boundsMetres"], work)
            terrain.update(render)
            for name in ("terrain-height.png", "terrain-lighting.png"):
                if name not in manifest["files"]:
                    manifest["files"].append(name)
            replacement = work / "package.zip"
            with zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
                for member in source.infolist():
                    if member.filename in ("manifest.json", "terrain.json", "terrain-height.png", "terrain-lighting.png"):
                        continue
                    with source.open(member) as input_file, output.open(member, "w") as output_file:
                        while chunk := input_file.read(1024 * 1024):
                            output_file.write(chunk)
                output.writestr("terrain.json", json.dumps(terrain, indent=2, sort_keys=True) + "\n")
                output.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                for name in ("terrain-height.png", "terrain-lighting.png"):
                    output.write(work / name, name)
        with zipfile.ZipFile(replacement) as check:
            if check.testzip() is not None:
                raise ValueError("Updated ZIP verification failed")
        digest = hashlib.sha256()
        with replacement.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
        size = replacement.stat().st_size
        os.replace(replacement, path)
    with sqlite3.connect(data / "jobs.sqlite3") as database:
        database.execute("UPDATE artifacts SET size_bytes=?,sha256=?,updated_at=? "
                         "WHERE job_id=? AND kind='map' AND state='ready'",
                         (size, digest.hexdigest(), time.time(), job_id))
    print(f"Updated {path} ({size} bytes, SHA256 {digest.hexdigest()})")


if __name__ == "__main__":
    upgrade(sys.argv[1])
