"""Locate and cache Slovenia CLSS source tiles."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import math
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlparse

from filelock import FileLock
import laspy
import rasterio
import requests
from rasterio.warp import transform

CRS = "EPSG:3794"
API = "https://lift.clss.si"
LAYER = "e41558a2-adcb-11ef-ad34-02420a000587"
MAP_ID = "0702ae28-a192-11ef-bf1e-0242ac120026"
# Public guest client parameters used by the CLSS web viewer. Override if changed.
GUEST_SECRET = os.environ.get("CLSS_GUEST_CLIENT_SECRET", "pfyQ8DqAPDU45vGaajkhfahjkr45t3rdasrFMuNFJkKmL2")


class SourceError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _Href(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")


def bounds_for(center: dict, width: float, height: float) -> dict:
    x, y = transform("EPSG:4326", CRS, [center["lon"]], [center["lat"]])
    return {"left": x[0] - width / 2, "bottom": y[0] - height / 2,
            "right": x[0] + width / 2, "top": y[0] + height / 2}


def cells_for(bounds: dict) -> list[tuple[int, int]]:
    return [(x, y)
            for x in range(math.floor(bounds["left"] / 1000), math.ceil(bounds["right"] / 1000))
            for y in range(math.floor(bounds["bottom"] / 1000), math.ceil(bounds["top"] / 1000))]


def _request(session, method, url, **kwargs):
    last = None
    for attempt in range(3):
        try:
            response = session.request(method, url, timeout=25, **kwargs)
            if response.status_code in (429, 500, 502, 503, 504):
                last = SourceError("SOURCE_UNAVAILABLE", f"CLSS returned HTTP {response.status_code}")
            else:
                response.raise_for_status()
                return response
        except requests.RequestException as error:
            last = SourceError("SOURCE_UNAVAILABLE", f"CLSS request failed: {error}")
        if attempt < 2:
            time.sleep(1 + 2 * attempt)
    raise last


def resolve(bounds: dict, products: tuple[str, ...]) -> list[dict]:
    """Ask the public CLSS grid for each intersecting 1 km tile's actual URLs."""
    session = requests.Session()
    token = _request(session, "POST", API + "/oauth/token", json={
        "grant_type": "password", "client_id": 0, "client_secret": GUEST_SECRET,
        "username": "__guest__", "password": "__guest__",
    }).json().get("access_token")
    if not token:
        raise SourceError("SOURCE_AUTH", "CLSS guest access did not return a token")
    headers = {"Authorization": "Bearer " + token, "mapid": MAP_ID,
               "device": "default", "x-requested-with": "XMLHttpRequest"}
    result = []
    for x, y in cells_for(bounds):
        cell = f"{x}_{y}"
        payload = {"attribute_field_names": ["ti_name", "datum_za_1", "link_pof", "link_dmr", "link_gkot"],
                   "offset": 0, "limit": 2, "buffer": 0, "source_srid": 3794,
                   "wkt": f"POINT({x * 1000 + 500} {y * 1000 + 500})"}
        rows = _request(session, "POST", f"{API}/xhr/layer/{LAYER}/data",
                        json=payload, headers=headers).json().get("data", [])
        matches = [row for row in rows if row.get("ti_name") == cell]
        if len(matches) != 1:
            raise SourceError("SOURCE_MISSING", f"CLSS tile {cell} is unavailable or ambiguous")
        row = matches[0]
        links = {}
        for product in products:
            parser = _Href()
            parser.feed(row.get("link_" + product) or "")
            url = parser.href
            parsed = urlparse(url or "")
            expected = f"{product.upper()}_{cell}." if product != "pof" else f"POF_{cell}."
            if (parsed.scheme != "https" or parsed.hostname != "assets.flycom.si"
                    or not parsed.path.startswith("/clss/raw/")
                    or not Path(parsed.path).name.startswith(expected)
                    or Path(parsed.path).suffix.lower() != (".tif" if product == "pof" else ".laz")):
                raise SourceError("SOURCE_MISSING", f"CLSS {product.upper()} is unavailable for tile {cell}")
            links[product] = url
        result.append({"cell": cell, "x": x, "y": y,
                       "acquired": row.get("datum_za_1"), "urls": links})
    return result


def _validate(path: Path, product: str, tile: dict):
    if product == "pof":
        with rasterio.open(path) as source:
            expected_bounds = (tile["x"] * 1000, tile["y"] * 1000,
                               (tile["x"] + 1) * 1000, (tile["y"] + 1) * 1000)
            if (source.crs is None or source.crs.to_epsg() != 3794 or source.count != 3
                    or source.width != 6250 or source.height != 6250
                    or source.dtypes != ("uint8", "uint8", "uint8")
                    or tuple(round(v, 3) for v in source.bounds) != expected_bounds):
                raise SourceError("SOURCE_INVALID", f"POF tile {tile['cell']} has unexpected geometry")
    else:
        with laspy.open(path) as source:
            header = source.header
            if (header.point_count <= 0 or header.parse_crs() is None
                    or header.parse_crs().to_epsg() != 3794
                    or header.x_min < tile["x"] * 1000 - 1
                    or header.x_max > (tile["x"] + 1) * 1000 + 1
                    or header.y_min < tile["y"] * 1000 - 1
                    or header.y_max > (tile["y"] + 1) * 1000 + 1):
                raise SourceError("SOURCE_INVALID", f"{product.upper()} tile {tile['cell']} has unexpected geometry")


def cached_download(tile: dict, product: str, cache_dir: Path, progress=None, cancelled=None) -> Path:
    """Resume static CLSS files safely and only expose validated completed files."""
    url = tile["urls"][product]
    filename = Path(urlparse(url).path).name
    destination = cache_dir / product / (hashlib.sha256(url.encode()).hexdigest()[:12] + "-" + filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        return _download_locked(tile, product, url, destination, progress, cancelled)


def _download_locked(tile, product, url, destination, progress, cancelled):
    if destination.is_file():
        try:
            _validate(destination, product, tile)
            return destination
        except Exception:
            destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".part")
    last = None
    for attempt in range(4):
        if cancelled and cancelled():
            raise SourceError("CANCELLED", "Job was deleted")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "HiresServer/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 45)) as response:
                if offset and response.status_code == 416:
                    match = re.fullmatch(r"bytes \*/(\d+)", response.headers.get("Content-Range", ""))
                    if match and int(match[1]) == offset:
                        _validate(partial, product, tile)
                        os.replace(partial, destination)
                        return destination
                    partial.unlink(missing_ok=True)
                    continue
                if response.status_code == 429:
                    raise SourceError("SOURCE_LIMIT", "CLSS download limit was reached")
                if response.status_code not in (200, 206):
                    raise SourceError("SOURCE_DOWNLOAD", f"CLSS {product.upper()} tile {tile['cell']} returned HTTP {response.status_code}")
                if offset and response.status_code != 206:
                    partial.unlink(missing_ok=True)
                    continue
                total = int(response.headers.get("Content-Length", "0")) + offset
                if offset:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset or int(match[3]) != total:
                        partial.unlink(missing_ok=True)
                        raise SourceError("SOURCE_DOWNLOAD", "Invalid resumed download range")
                if total <= 0:
                    raise SourceError("SOURCE_DOWNLOAD", "CLSS omitted the source file size")
                reserve = 512 * 1024 * 1024
                if shutil.disk_usage(destination.parent).free < total - offset + reserve:
                    raise SourceError("DISK_FULL", "Insufficient disk space for CLSS source download")
                with partial.open("ab" if offset else "wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if cancelled and cancelled():
                            raise SourceError("CANCELLED", "Job was deleted")
                        if chunk:
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if partial.stat().st_size != total:
                    raise SourceError("SOURCE_DOWNLOAD", f"Incomplete {product.upper()} tile {tile['cell']}")
            _validate(partial, product, tile)
            os.replace(partial, destination)
            if progress:
                progress(destination.stat().st_size)
            return destination
        except (requests.RequestException, OSError, SourceError) as error:
            last = error
            if isinstance(error, SourceError) and error.code in {"CANCELLED", "SOURCE_INVALID", "SOURCE_LIMIT", "DISK_FULL"}:
                raise
            if attempt < 3:
                time.sleep(1 + 2 * attempt)
    raise SourceError("SOURCE_DOWNLOAD", f"Could not download {product.upper()} tile {tile['cell']}: {last}")
