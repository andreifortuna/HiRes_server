"""Fast offline ETC2 RGB compression and validated KTX1 output."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from PIL import Image


ETCPAK_VERSION = "0.9.15"
KTX_SIGNATURE = b"\xabKTX 11\xbb\r\n\x1a\n"
KTX_HEADER = struct.Struct("<12s13I")


def read_ktx(path: Path, width: int, height: int, levels: int) -> list[bytes]:
    """Reject wrong formats, dimensions, incomplete chains, and truncated/trailing data."""
    data = path.read_bytes()
    if len(data) < KTX_HEADER.size:
        raise RuntimeError(f"truncated KTX header: {path}")
    header = KTX_HEADER.unpack_from(data)
    expected = (KTX_SIGNATURE, 0x04030201, 0, 1, 0, 0x9274, 0x1907,
                width, height, 0, 0, 1, levels)
    if header[:-1] != expected or header[-1] % 4:
        raise RuntimeError(f"invalid ETC2 RGB KTX header: {path}")
    offset = KTX_HEADER.size + header[-1]
    payloads = []
    for _ in range(levels):
        if offset + 4 > len(data):
            raise RuntimeError(f"missing KTX mip level: {path}")
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        expected_size = ((width + 3) // 4) * ((height + 3) // 4) * 8
        if size != expected_size or offset + size > len(data):
            raise RuntimeError(f"invalid ETC2 mip payload: {path}")
        payloads.append(data[offset:offset + size])
        offset += size
        width, height = max(1, width // 2), max(1, height // 2)
    if offset != len(data):
        raise RuntimeError(f"unexpected bytes after KTX mip levels: {path}")
    return payloads


def encode_texture(pixels, destination: Path) -> int:
    image = Image.fromarray(pixels)
    width, height = image.size
    if image.mode != "RGB" or max(width, height) > 2048:
        raise RuntimeError("ETC2 orthophoto tiles must be RGB and at most 2048 pixels per side")
    payloads = []
    # Generate every mip explicitly, including rectangular tails through 1x1.
    while True:
        import etcpak

        rgba = np.asarray(image.convert("RGBA"))
        # ETC2 stores complete 4x4 blocks even for 2x2 and 1x1 mip levels.
        if image.width % 4 or image.height % 4:
            rgba = np.pad(rgba, ((0, -image.height % 4), (0, -image.width % 4), (0, 0)),
                          mode="edge")
        payload = etcpak.compress_etc2_rgb(rgba.tobytes(), rgba.shape[1], rgba.shape[0])
        expected = ((image.width + 3) // 4) * ((image.height + 3) // 4) * 8
        if len(payload) != expected:
            raise RuntimeError("etcpak returned an invalid ETC2 block count")
        payloads.append(payload)
        if image.size == (1, 1):
            break
        image = image.resize((max(1, image.width // 2), max(1, image.height // 2)),
                             Image.Resampling.BOX)
    with destination.open("wb") as output:
        output.write(KTX_HEADER.pack(KTX_SIGNATURE, 0x04030201, 0, 1, 0, 0x9274, 0x1907,
                                     width, height, 0, 0, 1, len(payloads), 0))
        for payload in payloads:
            output.write(struct.pack("<I", len(payload)))
            output.write(payload)
    read_ktx(destination, width, height, len(payloads))
    return len(payloads)
