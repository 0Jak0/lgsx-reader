"""Leica's PropertyContainer: the key/value blobs in an LGSX project database
(camera intrinsics and poses, track metadata) and the file's ProjectInfo.

Layout: u16 count, then per entry u16 key length, key (NUL-terminated), u16
value length, two flag bytes, u8 type, value. Types: 0x0B f64, 0x08 i64,
0x00 string, 1/4/6/7 unsigned integers of the value's length; anything else
is kept as bytes.
"""

from __future__ import annotations

import struct


def parse_props(data: bytes) -> dict:
    """Parse a PropertyContainer into a dict (file order kept).

    Raises ValueError if a length runs past the end of ``data``.
    """
    out: dict = {}
    if not data:
        return out
    p = 0

    def take(n: int) -> bytes:
        nonlocal p
        if p + n > len(data):
            raise ValueError(f"property container cut short: {n} bytes wanted at {p} of {len(data)}")
        b = data[p : p + n]
        p += n
        return b

    (count,) = struct.unpack("<H", take(2))
    for _ in range(count):
        (klen,) = struct.unpack("<H", take(2))
        key = take(klen).rstrip(b"\0").decode("latin1")
        (vlen,) = struct.unpack("<H", take(2))
        kind = take(3)[2]
        raw = take(vlen)
        if kind == 0x0B and vlen == 8:
            value = struct.unpack("<d", raw)[0]
        elif kind == 0x08 and vlen == 8:
            value = struct.unpack("<q", raw)[0]
        elif kind == 0x00:
            value = raw.rstrip(b"\0").decode("latin1")
        elif kind in (1, 4, 6, 7) and 1 <= vlen <= 8:
            value = int.from_bytes(raw, "little")
        else:
            value = raw
        out[key] = value
    return out
