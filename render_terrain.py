"""Runtime mesh height and lighting assets for the Cocos terrain renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, Resampling
import rasterio


RENDER_SPACING_METRES = 0.5
LIGHTING_HEIGHT_SCALE = 1.0


def _shifted(values: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    # The lighting kernel reaches beyond a 1 km package. Extend boundary heights
    # for visual shading so the horizon and curvature terms stay continuous.
    rows = np.clip(np.arange(values.shape[0]) - dy, 0, values.shape[0] - 1)
    cols = np.clip(np.arange(values.shape[1]) - dx, 0, values.shape[1] - 1)
    return values[np.ix_(rows, cols)], np.ones(values.shape, dtype=bool)


def _lighting(heights: np.ndarray, sample_x: float, sample_y: float) -> Image.Image:
    """Pack HEDDA's sun visibility, horizon AO, relief, and curvature channels."""
    def offset(dx: int, dy: int, distance: int) -> tuple[int, int]:
        return round(dx * distance * 30.0 / sample_x), round(dy * distance * 30.0 / sample_y)

    def horizon_slope(dx: int, dy: int, distances) -> np.ndarray:
        horizon = np.zeros_like(heights)
        for distance in distances:
            ox, oy = offset(dx, dy, distance)
            sample, valid = _shifted(heights, -ox, -oy)
            ground_distance = np.hypot(ox * sample_x, oy * sample_y)
            slope = np.where(valid, (sample - heights) / ground_distance, -np.inf)
            horizon = np.maximum(horizon, slope)
        return horizon

    gradient_z, gradient_x = np.gradient(heights, sample_y, sample_x)
    nx, ny, nz = -gradient_x, np.ones_like(heights), -gradient_z
    inverse_length = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inverse_length, ny * inverse_length, nz * inverse_length

    def directional_light(horizontal_x: float, horizontal_z: float, altitude_degrees: float) -> np.ndarray:
        altitude = np.radians(altitude_degrees)
        horizontal_length = np.hypot(horizontal_x, horizontal_z)
        lx = np.cos(altitude) * horizontal_x / horizontal_length
        ly = np.sin(altitude)
        lz = np.cos(altitude) * horizontal_z / horizontal_length
        return np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

    sun_horizon = horizon_slope(-1, -1, range(1, 97))
    sun_visibility = np.clip((np.tan(np.radians(34.0)) + 0.10 - sun_horizon) / 0.20, 0.0, 1.0)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1))
    horizon_angles = [np.arctan(np.maximum(horizon_slope(dx, dy, (1, 2, 4, 8, 16, 32, 64, 96)), 0.0))
                      for dx, dy in directions]
    ambient_occlusion = np.clip(1.0 - np.mean(horizon_angles, axis=0) / 0.90, 0.42, 1.0)
    relief = (0.50 * directional_light(-1.0, -1.0, 42.0) +
              0.20 * directional_light(-1.0, 0.0, 52.0) +
              0.18 * directional_light(0.0, -1.0, 52.0) +
              0.12 * directional_light(1.0, 1.0, 60.0))
    relief = np.clip((relief - 0.18) / 0.72, 0.0, 1.0)

    def surrounding_mean(distance: int) -> np.ndarray:
        total = np.zeros_like(heights)
        count = np.zeros_like(heights)
        for dx, dy in directions:
            ox, oy = offset(dx, dy, distance)
            sample, valid = _shifted(heights, ox, oy)
            total += np.where(valid, sample, 0.0)
            count += valid
        return total / np.maximum(count, 1.0)

    curvature = (0.52 * (heights - surrounding_mean(2)) / 65.0 +
                 0.31 * (heights - surrounding_mean(6)) / 150.0 +
                 0.17 * (heights - surrounding_mean(16)) / 330.0)
    curvature -= np.median(curvature)
    curvature = 0.5 + 0.5 * np.tanh(curvature)
    packed = np.stack((sun_visibility, ambient_occlusion, relief, curvature), axis=2)
    return Image.fromarray(np.rint(packed * 255.0).astype(np.uint8), mode="RGBA")


def write_render_assets(heights: np.ndarray, bounds: dict, work: Path,
                        source_path: Path | None = None) -> dict:
    width_metres = bounds["right"] - bounds["left"]
    height_metres = bounds["top"] - bounds["bottom"]

    def dimension(length: float) -> int:
        cells = 1
        while length / cells > RENDER_SPACING_METRES and cells < 2048:
            cells *= 2
        return cells + 1

    width, height = dimension(width_metres), dimension(height_metres)
    if width * height > 4_200_000:
        raise ValueError("Runtime terrain mesh exceeds the renderer's supported size")
    spacing_x, spacing_y = width_metres / (width - 1), height_metres / (height - 1)
    rendered = np.full((height, width), np.nan, dtype=np.float32)
    destination = from_origin(bounds["left"] - spacing_x * 0.5,
                              bounds["top"] + spacing_y * 0.5, spacing_x, spacing_y)
    if source_path is None:
        reproject(heights, rendered,
                  src_transform=from_bounds(bounds["left"], bounds["bottom"], bounds["right"], bounds["top"],
                                            heights.shape[1], heights.shape[0]), src_crs="EPSG:3794",
                  dst_transform=destination, dst_crs="EPSG:3794", dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    else:
        with rasterio.open(source_path) as source:
            reproject(rasterio.band(source, 1), rendered,
                      dst_transform=destination, dst_crs="EPSG:3794", dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
    if not np.isfinite(rendered).all():
        raise ValueError("Runtime terrain has missing heights")
    minimum = min(float(heights.min()), float(rendered.min()))
    maximum = max(float(heights.max()), float(rendered.max()))
    maximum = max(maximum, minimum + 0.1)
    normalized = np.rint(np.clip((rendered - minimum) * 65535.0 / (maximum - minimum), 0, 65535)).astype(np.uint16)
    packed = np.empty((height, width, 3), dtype=np.uint8)
    packed[:, :, 0] = (normalized >> 8).astype(np.uint8)
    packed[:, :, 1] = (normalized & 255).astype(np.uint8)
    packed[:, :, 2] = 0
    Image.fromarray(packed, mode="RGB").save(
        work / "terrain-height.png", optimize=True)
    _lighting(rendered * LIGHTING_HEIGHT_SCALE, spacing_x, spacing_y).save(
        work / "terrain-lighting.png", optimize=True)
    return {
        "elevationMetres": {"minimum": minimum, "maximum": maximum},
        "renderHeight": {"file": "terrain-height.png", "width": width, "height": height,
                         "encoding": "RG16_UNORM",
                         "sampleSpacingMetres": {"x": spacing_x, "y": spacing_y}},
        "terrainLighting": {"file": "terrain-lighting.png", "width": width, "height": height,
                            "heightScale": LIGHTING_HEIGHT_SCALE,
                            "encoding": "RGBA PNG: sun visibility, ambient occlusion, relief, curvature"},
    }
