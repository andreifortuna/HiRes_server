"""Bake a restrained ground-normal signal from classified GKOT returns."""

from pathlib import Path

import laspy
import numpy as np
from PIL import Image


def _blur(values: np.ndarray) -> np.ndarray:
    weights = np.array([0.018, 0.091, 0.232, 0.318, 0.232, 0.091, 0.018], dtype=np.float32)
    padded = np.pad(values, ((0, 0), (3, 3)), mode="edge")
    horizontal = sum(weight * padded[:, i:i + values.shape[1]] for i, weight in enumerate(weights))
    padded = np.pad(horizontal, ((3, 3), (0, 0)), mode="edge")
    return sum(weight * padded[i:i + values.shape[0], :] for i, weight in enumerate(weights))


def write_gkot_normal(laz: Path, bounds: dict, output: Path, size: int = 1024) -> None:
    count = np.zeros((size, size), dtype=np.float32)
    heights = np.zeros_like(count)
    left, top = bounds["left"], bounds["top"]
    width, depth = bounds["right"] - left, top - bounds["bottom"]
    with laspy.open(laz) as source:
        for points in source.chunk_iterator(500_000):
            ground = np.asarray(points.classification) == 2
            if not np.any(ground):
                continue
            x = np.asarray(points.x)[ground]
            y = np.asarray(points.y)[ground]
            z = np.asarray(points.z)[ground]
            col = np.clip(((x - left) * size / width).astype(np.int32), 0, size - 1)
            row = np.clip(((top - y) * size / depth).astype(np.int32), 0, size - 1)
            index = row * size + col
            count += np.bincount(index, minlength=size * size).reshape(size, size)
            heights += np.bincount(index, weights=z, minlength=size * size).reshape(size, size)
    # Weighted blur suppresses gaps and point-level noise without flattening
    # terrain geometry. The signal affects illumination only, never elevation.
    weighted = _blur(heights)
    confidence = _blur(count)
    mean = weighted / np.maximum(confidence, 0.001)
    south, east = np.gradient(mean, depth / size, width / size)
    normal = np.stack((-east, np.ones_like(east), -south), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 0.001)
    # Octahedral encoding stores the horizontal components; green/y remains positive.
    denominator = np.abs(normal).sum(axis=-1)
    encoded = np.empty((size, size, 3), dtype=np.uint8)
    encoded[:, :, 0] = np.rint(np.clip(normal[:, :, 0] / denominator * 127.5 + 127.5, 0, 255)).astype(np.uint8)
    encoded[:, :, 1] = np.rint(np.clip(normal[:, :, 2] / denominator * 127.5 + 127.5, 0, 255)).astype(np.uint8)
    encoded[:, :, 2] = np.rint(np.clip(confidence / 4.0, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(encoded, "RGB").save(output, optimize=True)
