"""Low-frequency local colour and seamless rock detail for steep terrain."""

from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import rasterio

from sources import CRS


def write_cliff_material(bounds: dict, tiles: list[dict], sources: dict, work: Path) -> dict:
    size = 512
    transform = from_bounds(bounds["left"], bounds["bottom"], bounds["right"], bounds["top"], size, size)
    colour = np.zeros((size, size, 3), dtype=np.uint8)
    for tile in tiles:
        with rasterio.open(sources[(tile["cell"], "pof")]) as source:
            for band in range(3):
                reproject(rasterio.band(source, band + 1), colour[:, :, band],
                          dst_transform=transform, dst_crs=CRS, resampling=Resampling.average,
                          init_dest_nodata=False)

    # Periodic filtered noise stays seamless when sampled on a cliff face in world space.
    random = np.random.default_rng(1427).standard_normal((size, size))
    frequency = np.fft.fftfreq(size)
    squared = frequency[:, None] ** 2 + frequency[None, :] ** 2
    spectrum = np.power(squared + 0.0004, -0.55)
    spectrum[0, 0] = 0.0
    detail = np.fft.ifft2(np.fft.fft2(random) * spectrum).real
    detail = (detail - detail.mean()) / detail.std()
    alpha = np.rint(np.clip(0.5 + 0.22 * detail, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(colour, mode="RGB").save(work / "terrain-cliff-color.png", optimize=True)
    Image.fromarray(alpha, mode="L").save(work / "terrain-cliff-detail.png", optimize=True)
    return {"colourFile": "terrain-cliff-color.png", "detailFile": "terrain-cliff-detail.png",
            "baseColourResolution": size,
            "detailRepeatMetres": 64.0}
