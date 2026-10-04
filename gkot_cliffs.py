"""Conservative, near-view cliff faces supported by vertically separated GKOT ground returns."""

from pathlib import Path
import math

import laspy
import numpy as np


MAX_CELLS = 1000  # Two triangles per cell; well below the 50,000 visible-triangle ceiling.


def write_cliff_patches(laz: Path, bounds: dict, output: Path) -> int:
    output = Path(output)
    actual_width = bounds["right"] - bounds["left"]
    actual_depth = bounds["top"] - bounds["bottom"]
    width = math.ceil(actual_width)
    depth = math.ceil(actual_depth)
    count = np.zeros(width * depth, dtype=np.int32)
    low = np.full(width * depth, np.inf, dtype=np.float32)
    high = np.full(width * depth, -np.inf, dtype=np.float32)

    def read_points():
        with laspy.open(laz) as source:
            for points in source.chunk_iterator(500_000):
                ground = np.asarray(points.classification) == 2
                x = np.asarray(points.x)[ground]
                y = np.asarray(points.y)[ground]
                z = np.asarray(points.z)[ground]
                col = np.clip((x - bounds["left"]).astype(np.int32), 0, width - 1)
                row = np.clip((bounds["top"] - y).astype(np.int32), 0, depth - 1)
                yield row * width + col, z

    for index, z in read_points():
        count += np.bincount(index, minlength=count.size).astype(np.int32)
        np.minimum.at(low, index, z)
        np.maximum.at(high, index, z)
    spread = high - low
    candidate = ((count >= 16) & (spread >= 8.0)).reshape(depth, width)
    support = np.zeros_like(candidate, dtype=np.uint8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        support[max(0, dy):depth + min(0, dy), max(0, dx):width + min(0, dx)] += \
            candidate[max(0, -dy):depth - max(0, dy), max(0, -dx):width - max(0, dx)]
    candidate &= support >= 2
    candidate = candidate.ravel()
    middle = np.zeros(count.size, np.int16)
    bottom = np.zeros(count.size, np.int16)
    upper = np.zeros(count.size, np.int16)
    for index, z in read_points():
        chosen = candidate[index]
        index, z = index[chosen], z[chosen]
        if not len(index):
            continue
        gap = spread[index]
        midpoint_low = low[index] + 0.30 * gap
        midpoint_high = high[index] - 0.30 * gap
        np.add.at(middle, index[(z > midpoint_low) & (z < midpoint_high)], 1)
        np.add.at(bottom, index[z <= low[index] + 1.0], 1)
        np.add.at(upper, index[z >= high[index] - 1.0], 1)
    valid = np.flatnonzero(candidate & (middle >= 2) & (bottom >= 2) & (upper >= 2))
    remaining = set(map(int, valid))
    components = []
    while remaining:
        cluster = [remaining.pop()]
        for index in cluster:
            row, col = divmod(index, width)
            for adjacent in (index - width, index + width, index - 1, index + 1):
                if (adjacent == index - 1 and col == 0) or \
                   (adjacent == index + 1 and col == width - 1):
                    continue
                if adjacent in remaining:
                    remaining.remove(adjacent)
                    cluster.append(adjacent)
        if len(cluster) >= 12:
            components.append(cluster)
    components.sort(key=lambda cluster: sum(float(spread[i]) for i in cluster), reverse=True)
    selected = []
    for cluster in components:
        if len(selected) + len(cluster) <= MAX_CELLS:
            selected.extend(cluster)
    midpoint = np.where(count.reshape(depth, width) > 0,
                        0.5 * (np.nan_to_num(low.reshape(depth, width), posinf=0.0) +
                               np.nan_to_num(high.reshape(depth, width), neginf=0.0)), 0.0)
    with output.open("w", encoding="ascii") as mesh:
        mesh.write("# GKOT supported cliff faces; two triangles per face\n")
        face = 0
        for index in selected:
            row, col = divmod(int(index), width)
            west = midpoint[row, max(0, col - 1)]
            east = midpoint[row, min(width - 1, col + 1)]
            north = midpoint[max(0, row - 1), col]
            south = midpoint[min(depth - 1, row + 1), col]
            normal = np.array([west - east, north - south], dtype=np.float64)
            length = np.linalg.norm(normal)
            if not np.isfinite(length) or length < 0.1:
                continue
            normal /= length
            tangent = np.array([-normal[1], normal[0]]) * 0.48
            center = np.array([col + 0.5 - actual_width / 2,
                               row + 0.5 - actual_depth / 2]) + 0.35 * normal
            bottom_y, top_y = float(low[index]), float(high[index])
            corners = ((center - tangent, bottom_y), (center + tangent, bottom_y),
                       (center + tangent, top_y), (center - tangent, top_y))
            for horizontal, y in corners:
                mesh.write(f"v {horizontal[0]:.3f} {y:.3f} {horizontal[1]:.3f}\n")
            for horizontal, _ in corners:
                u = (horizontal[0] + actual_width / 2) / actual_width
                v = (horizontal[1] + actual_depth / 2) / actual_depth
                mesh.write(f"vt {u:.6f} {v:.6f}\n")
            mesh.write(f"vn {normal[0]:.5f} 0 {normal[1]:.5f}\n")
            first = face * 4 + 1
            n = face + 1
            mesh.write(f"f {first}/{first}/{n} {first+1}/{first+1}/{n} {first+2}/{first+2}/{n}\n")
            mesh.write(f"f {first}/{first}/{n} {first+2}/{first+2}/{n} {first+3}/{first+3}/{n}\n")
            face += 1
    return face * 2
