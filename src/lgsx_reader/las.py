"""A small streaming LAS 1.4 writer (point format 7: XYZ, intensity, return
1 of 1, class, point source ID, GPS time 0, 16-bit RGB), enough to hand an
LGSX's points to any LAS reader. The coordinate system goes in as an OGC WKT
VLR when known."""

from __future__ import annotations

import struct

import numpy as np

_HEADER = 375
_POINT = np.dtype(
    [
        ("x", "<i4"),
        ("y", "<i4"),
        ("z", "<i4"),
        ("intensity", "<u2"),
        ("returns", "u1"),
        ("flags", "u1"),
        ("classification", "u1"),
        ("user_data", "u1"),
        ("scan_angle", "<i2"),
        ("point_source_id", "<u2"),
        ("gps_time", "<f8"),
        ("red", "<u2"),
        ("green", "<u2"),
        ("blue", "<u2"),
    ]
)
assert _POINT.itemsize == 36


class LasWriter:
    """Write points in batches; the header is completed on :meth:`close`.

    ``offset`` and ``scale`` (3 floats each) set the integer grid; give the
    LGSX point cloud's origin and scale and every point is stored exactly.
    """

    def __init__(self, path, offset, scale, wkt: str | None = None):
        self._f = open(path, "wb")
        self._offset = np.asarray(offset, float)
        self._scale = np.asarray(scale, float)
        self._vlr = b""
        if wkt:
            data = wkt.encode() + b"\0"
            self._vlr = struct.pack("<H16sHH32s", 0, b"LASF_Projection", 2112, len(data), b"OGC WKT") + data
        self._count = 0
        self._by_return = 0
        self._lo = np.full(3, np.inf)
        self._hi = np.full(3, -np.inf)
        self._f.write(b"\0" * (_HEADER + len(self._vlr)))

    def write(self, xyz, rgb=None, intensity=None, classification=None, point_source_id=None):
        """Append points: ``xyz`` (n x 3 world), the rest optional (n,) /
        (n x 3 uint8 colour)."""
        xyz = np.asarray(xyz, float)
        n = len(xyz)
        if n == 0:
            return
        rec = np.zeros(n, _POINT)
        q = np.round((xyz - self._offset) / self._scale).astype(np.int64)
        rec["x"], rec["y"], rec["z"] = q[:, 0], q[:, 1], q[:, 2]
        rec["returns"] = 0x11  # return 1 of 1
        if intensity is not None:
            rec["intensity"] = intensity
        if classification is not None:
            rec["classification"] = np.minimum(np.asarray(classification), 255)
        if point_source_id is not None:
            rec["point_source_id"] = point_source_id
        if rgb is not None:
            c = np.asarray(rgb, np.uint16) * 257
            rec["red"], rec["green"], rec["blue"] = c[:, 0], c[:, 1], c[:, 2]
        self._f.write(rec.tobytes())
        self._count += n
        w = q * self._scale + self._offset
        self._lo = np.minimum(self._lo, w.min(axis=0))
        self._hi = np.maximum(self._hi, w.max(axis=0))

    def close(self):
        h = bytearray(_HEADER)
        struct.pack_into("<4s", h, 0, b"LASF")
        struct.pack_into("<H", h, 6, 0x10 if self._vlr else 0)  # global encoding: WKT
        struct.pack_into("<BB", h, 24, 1, 4)
        struct.pack_into("<32s32s", h, 26, b"lgsx-reader", b"lgsx-reader")
        struct.pack_into("<HH", h, 90, 1, 2026)
        struct.pack_into("<HIIBHI", h, 94, _HEADER, _HEADER + len(self._vlr), 1 if self._vlr else 0, 7, _POINT.itemsize, 0)
        struct.pack_into("<3d3d", h, 131, *self._scale, *self._offset)
        lo = np.where(np.isfinite(self._lo), self._lo, 0.0)
        hi = np.where(np.isfinite(self._hi), self._hi, 0.0)
        struct.pack_into("<6d", h, 179, hi[0], lo[0], hi[1], lo[1], hi[2], lo[2])
        struct.pack_into("<QQIQ", h, 227, 0, 0, 0, self._count)  # waveform, first EVLR, EVLR count, points
        struct.pack_into("<Q", h, 255, self._count)  # points by return: all first returns
        self._f.seek(0)
        self._f.write(bytes(h))
        self._f.write(self._vlr)
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
