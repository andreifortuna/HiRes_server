"""Build independently downloadable orthophoto/terrain and GKOT archives."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

import laspy
import lerc
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

from cliff_material import write_cliff_material
from gkot_detail import write_gkot_normal
from gkot_cliffs import write_cliff_patches
from render_terrain import write_render_assets
from sources import CRS, SourceError, cached_download, resolve

ORTHO_RESOLUTION = 0.16
HEIGHT_RESOLUTION = 0.5
KTX_CONTENT = 2046
ATTRIBUTION = (
    "Source: Surveying and Mapping Authority of the Republic of Slovenia (GURS).\n"
    "CLSS 2023-2025 POF aerial orthophoto, DMR elevation, and/or GKOT classified point cloud.\n"
    "Acquisition dates and source tiles are recorded in manifest.json.\n"
    "Licence: Creative Commons Attribution 4.0 International.\n"
    "https://www.e-prostor.gov.si/en/access-to-geodetic-data/\n"
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive(destination: Path, files: list[tuple[Path, str]]) -> tuple[int, str]:
    temporary = destination.with_suffix(".zip.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
            for path, name in files:
                if not path.is_file() or name.startswith("/") or ".." in Path(name).parts:
                    raise RuntimeError(f"Invalid package member: {name}")
                output.write(path, name)
        with zipfile.ZipFile(temporary) as result:
            if result.testzip() is not None or len(result.namelist()) != len(files):
                raise RuntimeError("Archive validation failed")
        os.replace(temporary, destination)
        return destination.stat().st_size, sha256(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _height_tile(path: Path, tile: dict) -> np.ndarray:
    grid = np.full((2000, 2000), np.nan, dtype=np.float32)
    seen = np.zeros((2000, 2000), dtype=np.bool_)
    with laspy.open(path) as source:
        for points in source.chunk_iterator(500_000):
            x, y, z = np.asarray(points.x), np.asarray(points.y), np.asarray(points.z)
            cols_float = (x - (tile["x"] * 1000 + 0.25)) * 2
            rows_float = ((tile["y"] + 1) * 1000 - 0.25 - y) * 2
            cols = np.rint(cols_float).astype(np.int32)
            rows = np.rint(rows_float).astype(np.int32)
            if (np.any(cols < 0) or np.any(cols >= 2000) or np.any(rows < 0)
                    or np.any(rows >= 2000) or np.max(np.abs(cols - cols_float)) > 0.01
                    or np.max(np.abs(rows - rows_float)) > 0.01
                    or not np.isfinite(z).all() or np.any(seen[rows, cols])):
                raise SourceError("SOURCE_INVALID", f"DMR tile {tile['cell']} has invalid samples")
            grid[rows, cols] = z.astype(np.float32)
            seen[rows, cols] = True
    return grid


def _encode_lerc(heights: np.ndarray) -> tuple[bytes, float]:
    heights = np.ascontiguousarray(heights.astype(np.float32, copy=False))
    # Leave rounding headroom in the 0.1 m bound, then verify the actual decoded data.
    bound = 0.099
    status, required = lerc.encode(heights, 1, False, None, bound, 0)
    if status != 0 or required <= 0:
        raise RuntimeError("LERC could not size the terrain")
    status, count, blob = lerc.encode(heights, 1, False, None, bound, required)
    if status != 0:
        raise RuntimeError("LERC could not encode the terrain")
    encoded = bytes(blob[:count])
    decoded = lerc.decode(encoded)
    if decoded[0] != 0 or decoded[1].shape != heights.shape:
        raise RuntimeError("LERC terrain verification failed")
    error = float(np.max(np.abs(decoded[1] - heights)))
    if error > 0.1:
        raise RuntimeError(f"LERC terrain error exceeds 0.1 m: {error}")
    return encoded, error


def build_terrain(work: Path, bounds: dict, tiles: list[dict], sources: dict) -> dict:
    xs = [t["x"] for t in tiles]
    ys = [t["y"] for t in tiles]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    mosaic = work / "dmr-mosaic.tif"
    transform = from_origin(min_x * 1000, (max_y + 1) * 1000, 0.5, 0.5)
    with rasterio.open(mosaic, "w", driver="GTiff", count=1, dtype="float32", crs=CRS,
                       width=(max_x - min_x + 1) * 2000, height=(max_y - min_y + 1) * 2000,
                       transform=transform, nodata=np.nan, BIGTIFF="IF_SAFER") as output:
        for tile in tiles:
            grid = _height_tile(sources[(tile["cell"], "dmr")], tile)
            row = (max_y - tile["y"]) * 2000
            col = (tile["x"] - min_x) * 2000
            output.write(grid, 1, window=Window(col, row, 2000, 2000))
    width = math.ceil((bounds["right"] - bounds["left"]) / HEIGHT_RESOLUTION)
    height = math.ceil((bounds["top"] - bounds["bottom"]) / HEIGHT_RESOLUTION)
    if width * height > 100_000_000:
        raise RuntimeError("Requested height grid exceeds the builder's memory limit")
    target_transform = from_bounds(bounds["left"], bounds["bottom"], bounds["right"],
                                   bounds["top"], width, height)
    heights = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(mosaic) as source:
        reproject(source=rasterio.band(source, 1), destination=heights,
                  dst_transform=target_transform, dst_crs=CRS, dst_nodata=np.nan,
                  resampling=Resampling.bilinear, num_threads=2)
    if not np.isfinite(heights).all():
        raise SourceError("SOURCE_MISSING", "DMR does not completely cover the requested rectangle")
    payload, error = _encode_lerc(heights)
    (work / "terrain-height.lerc").write_bytes(payload)
    render = write_render_assets(heights, bounds, work, mosaic)
    metadata = {
        "source": "CLSS 2023-2025 DMR", "crs": CRS, "boundsMetres": bounds,
        "heightFile": "terrain-height.lerc", "encoding": "LERC Float32, north-to-south",
        "grid": {"width": width, "height": height,
                 "pixelSizeMetres": {"x": (bounds["right"]-bounds["left"])/width,
                                      "y": (bounds["top"]-bounds["bottom"])/height},
                 "samplesAtPixelCenters": True},
        "elevationMetres": {"minimum": float(heights.min()), "maximum": float(heights.max())},
        "maxEncodingErrorMetres": error,
        "sourceTiles": [t["cell"] for t in tiles],
        **render,
    }
    write_json(work / "terrain.json", metadata)
    mosaic.unlink(missing_ok=True)
    return metadata


def _texture_tile(pixels: np.ndarray, destination: Path) -> tuple[int, int, int]:
    valid_h, valid_w = pixels.shape[:2]
    padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
    texture_w = 1 << (valid_w + 1).bit_length()
    texture_h = 1 << (valid_h + 1).bit_length()
    padded = np.pad(padded, ((0, texture_h - padded.shape[0]),
                             (0, texture_w - padded.shape[1]), (0, 0)), mode="edge")
    Image.fromarray(padded, "RGB").save(destination, optimize=True)
    return texture_w, texture_h, int(math.log2(max(texture_w, texture_h))) + 1


def build_imagery(work: Path, bounds: dict, tiles: list[dict], sources: dict, progress) -> dict:
    width = math.ceil((bounds["right"] - bounds["left"]) / ORTHO_RESOLUTION)
    height = math.ceil((bounds["top"] - bounds["bottom"]) / ORTHO_RESOLUTION)
    entries = []
    total = math.ceil(width / KTX_CONTENT) * math.ceil(height / KTX_CONTENT)
    for row in range(0, height, KTX_CONTENT):
        for col in range(0, width, KTX_CONTENT):
            valid_w = min(KTX_CONTENT, width - col)
            valid_h = min(KTX_CONTENT, height - row)
            tile_bounds = (bounds["left"] + col * (bounds["right"] - bounds["left"]) / width,
                           bounds["top"] - (row + valid_h) * (bounds["top"] - bounds["bottom"]) / height,
                           bounds["left"] + (col + valid_w) * (bounds["right"] - bounds["left"]) / width,
                           bounds["top"] - row * (bounds["top"] - bounds["bottom"]) / height)
            transform = from_bounds(*tile_bounds, valid_w, valid_h)
            pixels = np.zeros((3, valid_h, valid_w), dtype=np.uint8)
            coverage = np.zeros((valid_h, valid_w), dtype=np.uint8)
            for tile in tiles:
                with rasterio.open(sources[(tile["cell"], "pof")]) as source:
                    for band in (1, 2, 3):
                        reproject(rasterio.band(source, band), pixels[band - 1],
                                  dst_transform=transform, dst_crs=CRS,
                                  resampling=Resampling.bilinear, init_dest_nodata=False)
                    reproject(source.read_masks(1), coverage,
                              src_transform=source.transform, src_crs=source.crs,
                              dst_transform=transform, dst_crs=CRS,
                              resampling=Resampling.nearest, init_dest_nodata=False)
            if np.any(coverage == 0):
                raise SourceError("SOURCE_MISSING", "POF does not completely cover the requested rectangle")
            name = f"ortho-{col // KTX_CONTENT}-{row // KTX_CONTENT}.png"
            texture_w, texture_h, levels = _texture_tile(np.moveaxis(pixels, 0, 2), work / name)
            entries.append({"file": name, "x": col // KTX_CONTENT, "y": row // KTX_CONTENT,
                            "validWidth": valid_w, "validHeight": valid_h,
                            "contentOffset": 1, "textureWidth": texture_w,
                            "textureHeight": texture_h, "mipLevels": levels})
            progress("encoding_imagery", len(entries), total)
    metadata = {"source": "CLSS 2023-2025 POF aerial orthophoto", "crs": CRS,
                "boundsMetres": bounds, "width": width, "height": height,
                "pixelSizeMetres": {"x": (bounds["right"]-bounds["left"])/width,
                                     "y": (bounds["top"]-bounds["bottom"])/height},
                "encoding": "lossless RGB PNG; mipmaps generated by viewer", "tiles": entries,
                "sourceTiles": [t["cell"] for t in tiles]}
    write_json(work / "orthophoto.json", metadata)
    return metadata


def build_map(job: dict, work: Path, cache: Path, target: Path, progress, cancelled) -> tuple[int, str]:
    bounds = job["bounds"]
    progress("locating_sources", 0, 0)
    tiles = resolve(bounds, ("pof", "dmr"))
    sources = {}
    total = len(tiles) * 2
    for count, (tile, product) in enumerate(((t, p) for t in tiles for p in ("pof", "dmr")), 1):
        progress("downloading_sources", count - 1, total)
        sources[(tile["cell"], product)] = cached_download(tile, product, cache, cancelled=cancelled)
        progress("downloading_sources", count, total)
    if cancelled():
        raise SourceError("CANCELLED", "Job was deleted")
    progress("building_terrain", 0, len(tiles))
    build_terrain(work, bounds, tiles, sources)
    imagery = build_imagery(work, bounds, tiles, sources, progress)
    imagery["cliffMaterial"] = write_cliff_material(bounds, tiles, sources, work)
    write_json(work / "orthophoto.json", imagery)
    manifest = {"formatVersion": 1, "kind": "map", "id": job["id"],
                "centerWgs84": job["center"], "boundsMetres": bounds, "crs": CRS,
                "widthMeters": job["widthMeters"], "heightMeters": job["heightMeters"],
                "sourceTiles": [{"id": t["cell"], "acquired": t["acquired"]} for t in tiles],
                "files": ["terrain.json", "terrain-height.lerc", "terrain-height.png",
                          "terrain-lighting.png", "terrain-cliff-color.png",
                          "terrain-cliff-detail.png", "orthophoto.json"]
                         + [t["file"] for t in imagery["tiles"]]
                         + [f"source/dmr/{sources[(t['cell'], 'dmr')].name}" for t in tiles]}
    write_json(work / "manifest.json", manifest)
    (work / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    files = [(work / name, name) for name in manifest["files"] if not name.startswith("source/")]
    files += [(sources[(t["cell"], "dmr")], f"source/dmr/{sources[(t['cell'], 'dmr')].name}") for t in tiles]
    files += [(work / "manifest.json", "manifest.json"), (work / "ATTRIBUTION.txt", "ATTRIBUTION.txt")]
    if cancelled():
        raise SourceError("CANCELLED", "Job was deleted")
    progress("packaging", 0, len(files))
    return archive(target, files)


def build_gkot(job: dict, work: Path, cache: Path, target: Path, progress, cancelled) -> tuple[int, str]:
    bounds = job["bounds"]
    progress("locating_sources", 0, 0)
    tiles = resolve(bounds, ("gkot",))
    paths = []
    for count, tile in enumerate(tiles, 1):
        progress("downloading_sources", count - 1, len(tiles))
        paths.append(cached_download(tile, "gkot", cache, cancelled=cancelled))
        progress("downloading_sources", count, len(tiles))
    output = work / "gkot.laz"
    count = 0
    with laspy.open(paths[0]) as first:
        header = copy.deepcopy(first.header)
    with laspy.open(output, mode="w", header=header) as writer:
        for index, path in enumerate(paths, 1):
            progress("clipping_points", index - 1, len(paths))
            with laspy.open(path) as source:
                if source.header.point_format != header.point_format or source.header.scales.tolist() != header.scales.tolist() or source.header.offsets.tolist() != header.offsets.tolist():
                    raise SourceError("SOURCE_INVALID", f"GKOT tile {tiles[index-1]['cell']} uses incompatible LAS attributes")
                for points in source.chunk_iterator(500_000):
                    if cancelled():
                        raise SourceError("CANCELLED", "Job was deleted")
                    x, y = np.asarray(points.x), np.asarray(points.y)
                    chosen = ((x >= bounds["left"]) & (x < bounds["right"])
                              & (y >= bounds["bottom"]) & (y < bounds["top"]))
                    selected = points[chosen]
                    if len(selected):
                        writer.write_points(selected)
                        count += len(selected)
            progress("clipping_points", index, len(paths))
    if count == 0:
        raise SourceError("SOURCE_MISSING", "GKOT has no points within the requested rectangle")
    with laspy.open(output) as check:
        if check.header.point_count != count:
            raise RuntimeError("Cropped GKOT point count verification failed")
    progress("building_normal_detail", 0, 1)
    write_gkot_normal(output, bounds, work / "gkot-normal.png")
    cliff_triangles = write_cliff_patches(output, bounds, work / "gkot-cliffs.obj")
    progress("building_normal_detail", 1, 1)
    manifest = {"formatVersion": 1, "kind": "gkot", "id": job["id"],
                "centerWgs84": job["center"], "boundsMetres": bounds, "crs": CRS,
                "widthMeters": job["widthMeters"], "heightMeters": job["heightMeters"],
                "pointCount": count, "pointFormat": str(header.point_format),
                "cliffTriangles": cliff_triangles,
                "sourceTiles": [{"id": t["cell"], "acquired": t["acquired"]} for t in tiles],
                "files": ["gkot.laz", "gkot-normal.png", "gkot-cliffs.obj"]}
    write_json(work / "manifest.json", manifest)
    (work / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    if cancelled():
        raise SourceError("CANCELLED", "Job was deleted")
    progress("packaging", 0, 5)
    return archive(target, [(output, "gkot.laz"),
                            (work / "gkot-normal.png", "gkot-normal.png"),
                            (work / "gkot-cliffs.obj", "gkot-cliffs.obj"),
                            (work / "manifest.json", "manifest.json"),
                            (work / "ATTRIBUTION.txt", "ATTRIBUTION.txt")])
