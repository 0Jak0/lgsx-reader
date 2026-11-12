"""The point cloud: a Hexagon HSPC archive (``mMPK``) in the ``PointCloud``
entry, holding octree nodes (``APRN``), two trees (``APRT``), the
``hspcInfo.json`` header and sometimes a text note. Members are found by
their first bytes, as their order follows a name trie.

A node stores each attribute as zstd-compressed byte planes:

* every plane holds its points pair-swapped (element 2k is point 2k+1 and
  vice versa; an odd last element stays);
* POSITION (12 planes) is a 96-bit Morton code, delta-coded: prefix sum mod
  2**96, de-interleave (bit 3i -> x, 3i+1 -> y, 3i+2 -> z), subtract 2**31,
  then ``grid * scale + origin``;
* U16 / U8 attributes (2 / 1 planes) are delta-coded: prefix sum mod 2**16 /
  2**8;
* R, G and B share one 3-plane stream, placed after all the others: G is a
  prefix sum, B = plane 1 + G, R = plane 2 + G (mod 256).
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
import zstandard

FORMAT_VERSION = 22
_ARCHIVE_HEADER = 56
_NODE_ATTRS = 0xA8
_TYPE_PLANES = {10: 12, 1: 2, 0: 1}  # M96, U16, U8
_RGB_ROLES = (4, 5, 6)


@dataclass(frozen=True)
class Node:
    """One octree node: its member number, byte offset inside the
    ``PointCloud`` entry, size and point count."""

    member: int
    offset: int
    size: int
    point_count: int


@dataclass
class Header:
    """The point cloud's ``hspcInfo.json``."""

    point_count: int
    filtered_point_count: int
    origin: np.ndarray
    scale: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    attributes: list
    setup_labels: list
    raw: dict = field(repr=False)


class PointCloud:
    """An LGSX's point cloud. Get it from :meth:`lgsx_reader.Lgsx.point_cloud`."""

    def __init__(self, lgsx, entry):
        self._lgsx = lgsx
        self._entry = entry
        h = lgsx._read_in(entry, 0, _ARCHIVE_HEADER)
        if h[:4] != b"mMPK":
            raise ValueError("the point cloud is not an HSPC archive (no mMPK header)")
        ver, _, count, _, _, index_len, sizes_len = struct.unpack_from("<7I", h, 4)
        if ver != 1:
            raise ValueError(f"HSPC archive version {ver} is not supported")
        (data_offset,) = struct.unpack_from("<Q", h, 32)
        blob = lgsx._read_in(entry, _ARCHIVE_HEADER + index_len, sizes_len)
        (raw_len,) = struct.unpack_from("<I", blob, 0)
        sizes = np.frombuffer(zlib.decompress(blob[4:]), dtype="<u4")
        if len(sizes) * 4 != raw_len or len(sizes) != count:
            raise ValueError("archive size table does not match its member count")
        offsets = data_offset + np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)[:-1]])
        if int(offsets[-1] + sizes[-1]) != entry.size:
            raise ValueError("archive members do not end at the end of the entry")
        self.nodes: list[Node] = []
        header = None
        for k, (off, size) in enumerate(zip(offsets.tolist(), sizes.tolist())):
            head = lgsx._read_in(entry, off, min(24, size))
            if head[:4] == b"APRN":
                (n,) = struct.unpack_from("<I", head, 20)
                self.nodes.append(Node(k, off, size, n))
            elif head[:1] == b"{" and header is None:
                info = json.loads(lgsx._read_in(entry, off, size))
                header = info.get("tree-header-info")
        if header is None:
            raise ValueError("the point cloud has no hspcInfo.json header")
        if header["format-version"] != FORMAT_VERSION:
            raise ValueError(f"HSPC format version {header['format-version']} is not supported (this reader knows {FORMAT_VERSION})")
        labels: list = []
        for key, label in (header.get("metadata", {}).get("SetupLUT") or {}).items():
            if label == "EndofSetupIndices" or not key.isdigit():
                continue
            i = int(key)
            labels.extend([""] * (i + 1 - len(labels)))
            labels[i] = label
        bb = header["bounding-box"]
        self.header = Header(
            point_count=header["point-count"],
            filtered_point_count=header.get("filtered-point-count", 0),
            origin=np.array(header["origin"], float),
            scale=np.array(header["scale"], float),
            bounds_min=np.array(bb["min"], float),
            bounds_max=np.array(bb["max"], float),
            attributes=[(a["name"], a["type"]) for a in header["attributes"]],
            setup_labels=labels,
            raw=header,
        )
        total = sum(n.point_count for n in self.nodes)
        if total != self.header.point_count:
            raise ValueError(f"the nodes hold {total} points, the header says {self.header.point_count}")

    @property
    def point_count(self) -> int:
        return self.header.point_count

    def read_node(self, node: Node) -> dict:
        """Decode one node. Returns a dict of numpy arrays:

        ``xyz`` (n x 3 float64, world / project coordinates), ``grid`` (n x 3
        int64 on the cloud's grid) and, as the cloud has them, ``rgb`` (n x 3
        uint8), ``intensity``, ``classification``, ``setup`` (uint16: an index
        into :attr:`Header.setup_labels`), ``normal_code`` (uint16, see
        :func:`normals`) and any other attribute by its name in lower case.
        """
        return decode_node(self._lgsx._read_in(self._entry, node.offset, node.size))

    def iter_nodes(self):
        """Every node, decoded, in file order: ``(node, dict)``."""
        for node in self.nodes:
            yield node, self.read_node(node)


def _unswap(n: int) -> np.ndarray:
    idx = np.arange(n) ^ 1
    if n % 2:
        idx[-1] = n - 1
    return idx


def _compact(x: np.ndarray) -> np.ndarray:
    """Every third bit of each uint64 (bits 0, 3, 6, ...), packed."""
    u = np.uint64
    x = x & u(0x1249249249249249)
    x = (x ^ (x >> u(2))) & u(0x10C30C30C30C30C3)
    x = (x ^ (x >> u(4))) & u(0x100F00F00F00F00F)
    x = (x ^ (x >> u(8))) & u(0x001F0000FF0000FF)
    x = (x ^ (x >> u(16))) & u(0x001F00000000FFFF)
    x = (x ^ (x >> u(32))) & u(0x00000000001FFFFF)
    return x


def _positions(planes: list) -> np.ndarray:
    """12 planes of u96 Morton deltas -> grid coordinates (n x 3 int64)."""
    u = np.uint64
    m32 = u(0xFFFFFFFF)
    limb = [np.zeros(len(planes[0]), np.uint64) for _ in range(3)]
    for j, p in enumerate(planes):
        limb[j // 4] |= p.astype(np.uint64) << u(8 * (j % 4))
    # Prefix sum mod 2**96 in three 32-bit limbs; each cumulative sum stays
    # under 2**64 for fewer than 2**32 points.
    c0 = np.cumsum(limb[0], dtype=np.uint64)
    c1 = np.cumsum(limb[1], dtype=np.uint64) + (c0 >> u(32))
    c2 = np.cumsum(limb[2], dtype=np.uint64) + (c1 >> u(32))
    lo = (c0 & m32) | ((c1 & m32) << u(32))
    hi = ((c2 & m32) << u(1)) | (lo >> u(63))  # code bits 63..95
    lo = lo & u(0x7FFFFFFFFFFFFFFF)
    xyz = np.empty((len(lo), 3), np.int64)
    for axis in range(3):
        a = u(axis)
        v = _compact(lo >> a) | (_compact(hi >> a) << u(21))
        xyz[:, axis] = v.astype(np.int64) - (1 << 31)
    return xyz


def _running(planes: list) -> np.ndarray:
    if len(planes) == 2:
        v = planes[0].astype(np.uint64) | (planes[1].astype(np.uint64) << np.uint64(8))
        return (np.cumsum(v, dtype=np.uint64) & np.uint64(0xFFFF)).astype(np.uint16)
    return (np.cumsum(planes[0].astype(np.uint64), dtype=np.uint64) & np.uint64(0xFF)).astype(np.uint16)


def decode_node(b: bytes) -> dict:
    """Decode a whole ``APRN`` member (see :meth:`PointCloud.read_node`)."""
    if len(b) <= _NODE_ATTRS + 4 or b[:4] != b"APRN":
        raise ValueError("not a point cloud node")
    if struct.unpack_from("<I", b, 12)[0] != FORMAT_VERSION:
        raise ValueError("node format version is not supported")
    (n,) = struct.unpack_from("<I", b, 20)
    scale = np.array(struct.unpack_from("<3d", b, 0x3C))
    origin = np.array(struct.unpack_from("<3d", b, 0x54))
    (count,) = struct.unpack_from("<I", b, _NODE_ATTRS)
    r = _NODE_ATTRS + 4
    streams = []  # (target, planes)
    colour = 0
    for _ in range(count):
        ty, role, _z, nl = struct.unpack_from("<BBIH", b, r)
        r += 8
        name = b[r : r + nl].decode("latin1")
        r += nl
        if ty not in _TYPE_PLANES:
            raise ValueError(f"attribute {name} has type {ty}, which this reader does not know")
        if role in _RGB_ROLES:
            colour += 1
            continue
        if role == 0:
            target = "position"
        elif role == 8:
            target = "intensity"
        elif role == 9:
            target = "classification"
        elif name.startswith("SetupIndex"):
            target = "setup"
        elif name == "Normal" and ty == 1:
            target = "normal_code"
        else:
            target = name.lower()
        streams.append((target, _TYPE_PLANES[ty]))
    if colour not in (0, 3):
        raise ValueError(f"the node has {colour} of the R, G, B channels")
    if colour:
        streams.append(("rgb", 3))
    stream_count = b[r + 1]
    if stream_count != len(streams):
        raise ValueError(f"the node has {stream_count} streams for {len(streams)} attributes")
    offsets = struct.unpack_from(f"<{stream_count}Q", b, r + 2)
    base = r + 2 + 8 * stream_count
    at = base
    swap = _unswap(n)
    dz = zstandard.ZstdDecompressor()
    out: dict = {}
    for k, (target, nplanes) in enumerate(streams):
        if at - base != offsets[k]:
            raise ValueError(f"stream {k} starts at {at - base}, the table says {offsets[k]}")
        planes = []
        for _ in range(nplanes):
            (clen,) = struct.unpack_from("<Q", b, at)
            at += 8
            raw = dz.decompressobj().decompress(b[at : at + clen])
            if len(raw) != n:
                raise ValueError(f"plane holds {len(raw)} bytes for {n} points")
            planes.append(np.frombuffer(raw, np.uint8)[swap])
            at += clen
        if target == "position":
            out["grid"] = _positions(planes)
        elif target == "rgb":
            g = (np.cumsum(planes[0].astype(np.uint64)) & 0xFF).astype(np.uint8)
            out["rgb"] = np.stack([planes[2] + g, g, planes[1] + g], axis=1).astype(np.uint8)
        else:
            out[target] = _running(planes)
    if at != len(b):
        raise ValueError(f"{len(b) - at} bytes left after the last plane")
    out["xyz"] = out["grid"] * scale + origin
    return out


# ---- normals ----------------------------------------------------------------

_RINGS = 226
_THETA = np.arange(_RINGS) * np.pi / _RINGS
_BINS = np.maximum(np.ceil(2 * _RINGS * np.sin(_THETA)), 1).astype(np.int64)
_OFFSETS = np.concatenate([[0], np.cumsum(_BINS)])  # 65,143 codes


def normals(code: np.ndarray) -> np.ndarray:
    """Decode ``CodecNormal16PcApi`` normal codes to unit vectors (n x 3;
    NaN for 0xFFFF, no normal). Ring k = 0..225 sits at polar angle k*pi/226
    from +z with ceil(452 sin theta) azimuth bins; codes count ring after
    ring; within a ring code j points at azimuth 2*pi*(j + 1/2)/bins + pi.
    Normals face the scanner."""
    c = np.asarray(code, np.int64)
    ok = c < _OFFSETS[-1]
    ring = np.clip(np.searchsorted(_OFFSETS, c, side="right") - 1, 0, _RINGS - 1)
    th = _THETA[ring]
    phi = 2 * np.pi * (c - _OFFSETS[ring] + 0.5) / _BINS[ring] + np.pi
    out = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], axis=1)
    out[~ok] = np.nan
    return out
