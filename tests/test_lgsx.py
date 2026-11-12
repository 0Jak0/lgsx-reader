"""Tests: synthetic files for every part of the format, and (skipped unless
their paths are set in LGSX_MLS and LGSX_TLS) the two jobs the reader was
checked on, whose node checksums match an independent Rust reader's."""

import os
import sqlite3
import struct
import tempfile

import numpy as np
import pytest
import zstandard

import lgsx_reader
from lgsx_reader.las import LasWriter
from lgsx_reader.points import _OFFSETS, decode_node, normals

MLS = os.environ.get("LGSX_MLS", "")
TLS = os.environ.get("LGSX_TLS", "")


def morton(x, y, z):
    c = 0
    for b in range(32):
        c |= ((x >> b) & 1) << (3 * b) | ((y >> b) & 1) << (3 * b + 1) | ((z >> b) & 1) << (3 * b + 2)
    return c


def swap(plane):
    n = len(plane)
    idx = [i ^ 1 if (i ^ 1) < n else i for i in range(n)]
    return bytes(plane[i] for i in idx)


def synthetic_node(pts, intensity, setup, rgb):
    """A node encoded the way the files do it (grid ints biased by 2**31)."""
    n = len(pts)
    b = bytearray(0xA8)
    b[:4] = b"APRN"
    struct.pack_into("<I", b, 12, 22)
    struct.pack_into("<I", b, 20, n)
    struct.pack_into("<6d", b, 0x3C, 0.001, 0.001, 0.001, 100.0, 200.0, 300.0)
    attrs = [(10, 0, "POSITION"), (0, 4, "R"), (0, 5, "G"), (0, 6, "B"), (1, 8, "INTENSITY"), (0, 100, "SetupIndexU8")]
    b += struct.pack("<I", len(attrs))
    for ty, role, name in attrs:
        b += struct.pack("<BBIH", ty, role, 0, len(name)) + name.encode()
    prev = 0
    pos = [bytearray() for _ in range(12)]
    for p in pts:
        c = morton(*p)
        d = (c - prev) % (1 << 96)
        prev = c
        for j in range(12):
            pos[j].append((d >> (8 * j)) & 0xFF)
    prev = 0
    lo, hi = bytearray(), bytearray()
    for v in intensity:
        d = (v - prev) % 65536
        prev = v
        lo.append(d & 0xFF)
        hi.append(d >> 8)
    prev = 0
    sp = bytearray()
    for v in setup:
        sp.append((v - prev) % 256)
        prev = v
    g0, gp = bytearray(), 0
    bp, rp = bytearray(), bytearray()
    for r, g, bl in rgb:
        g0.append((g - gp) % 256)
        gp = g
        bp.append((bl - g) % 256)
        rp.append((r - g) % 256)
    streams = [pos, [lo, hi], [sp], [g0, bp, rp]]
    body = bytearray()
    offsets = []
    for planes in streams:
        offsets.append(len(body))
        for p in planes:
            z = zstandard.ZstdCompressor().compress(swap(p))
            body += struct.pack("<Q", len(z)) + z
    b += bytes([100, len(offsets)]) + struct.pack(f"<{len(offsets)}Q", *offsets) + body
    struct.pack_into("<I", b, 4, len(b))
    return bytes(b)


def test_decodes_a_synthetic_node():
    pts = [((1 << 31) + 5, (1 << 31) - 7, 1 << 31), ((1 << 31) + 6, (1 << 31) + 1000, (1 << 31) + 3), (0, (1 << 32) - 1, 12)]
    rgb = [(10, 20, 30), (255, 0, 128), (1, 2, 3)]
    d = decode_node(synthetic_node(pts, [100, 65000, 7], [3, 1, 250], rgb))
    assert d["grid"].tolist() == [[5, -7, 0], [6, 1000, 3], [-(1 << 31), (1 << 31) - 1, 12 - (1 << 31)]]
    assert d["intensity"].tolist() == [100, 65000, 7]
    assert d["setup"].tolist() == [3, 1, 250]
    assert d["rgb"].tolist() == [list(c) for c in rgb]
    assert np.allclose(d["xyz"][1], [100.006, 201.0, 300.003])


def test_morton_over_many_points():
    rng = np.random.default_rng(1)
    pts = [tuple(int(v) for v in rng.integers(0, 1 << 32, 3)) for _ in range(301)]
    d = decode_node(synthetic_node(pts, [0] * 301, [0] * 301, [(0, 0, 0)] * 301))
    want = np.array(pts, np.int64) - (1 << 31)
    assert (d["grid"] == want).all()


def test_normal_table():
    assert _OFFSETS[-1] == 65143
    up = normals(np.array([0]))[0]
    assert np.allclose(up, [0, 0, 1])
    assert np.isnan(normals(np.array([0xFFFF]))).all()
    n = normals(np.arange(0, 65143, 97))
    assert np.allclose(np.linalg.norm(n, axis=1), 1)


def test_props():
    def entry(key, kind, value):
        k = key.encode() + b"\0"
        return struct.pack("<H", len(k)) + k + struct.pack("<H", len(value)) + bytes([1, 1, kind]) + value

    data = struct.pack("<H", 3) + entry("Field_2", 0x0B, struct.pack("<d", 3590.5)) + entry("ImgW", 7, struct.pack("<I", 4096)) + entry("Model", 0, b"TRK100\0")
    assert lgsx_reader.parse_props(data) == {"Field_2": 3590.5, "ImgW": 4096, "Model": "TRK100"}
    with pytest.raises(ValueError):
        lgsx_reader.parse_props(data[:-3])


def container(entries):
    out = bytearray(b"wild::geosystems")
    placed = []
    for name, data in entries:
        placed.append((name, len(out), len(data)))
        out += data
    heads = []
    for name, off, size in placed:
        heads.append(len(out))
        n0 = name.encode() + b"\0"
        dg = b"00\0"
        hlen = 12 + len(n0) + 17 + 2 + len(dg)
        out += struct.pack("<HiIH", hlen, -1, 0, len(n0)) + n0 + bytes([1]) + struct.pack("<QQ", off, size) + struct.pack("<H", len(dg)) + dg
    d = len(out)
    out += struct.pack("<I", len(heads)) + b"".join(struct.pack("<Q", h) for h in heads) + b"footer" + struct.pack("<Q", d)
    return bytes(out)


def test_container_and_project(tmp_path):
    db = tmp_path / "p.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        """
        create table Hds_OdbSetup(mId, mLabel, mSetupIndex, mResourceId, mPosition0, mPosition1, mPosition2, mRotation0, mRotation1, mRotation2, mRotation3);
        insert into Hds_OdbSetup values (1, 'Setup 001', 0, 'abc', 10.0, 20.0, 1.5, 0, 0, 0, 1);
        create table Hds_OdbCoordSystem(mId, mName, mWKT);
        insert into Hds_OdbCoordSystem values (1, 'MGA55_20_TAS', 'PROJCS["MGA55_20_TAS"]');
        """
    )
    con.commit()
    con.close()
    info = struct.pack("<HH", 4, 12) + b"ProjectInfo\0" + b"\0\0" + struct.pack("<H", 1) + struct.pack("<H", 12) + b"ProjectName\0" + struct.pack("<H", 5) + bytes([1, 1, 0]) + b"Test\0"
    path = tmp_path / "t.lgsx"
    path.write_bytes(container([("", info), ("ProjectDatabase", db.read_bytes()), ("Hello", b"hi")]))
    with lgsx_reader.open(path) as f:
        assert [e.name for e in f.entries] == ["", "ProjectDatabase", "Hello"]
        assert f.read("Hello") == b"hi"
        assert f.info["ProjectName"] == "Test"
        p = f.project()
        assert p.name == "Test" and p.crs_name == "MGA55_20_TAS" and not p.is_mobile
        assert p.setups[0].label == "Setup 001" and np.allclose(p.setups[0].position, [10, 20, 1.5])
    bad = tmp_path / "bad.lgsx"
    bad.write_bytes(b"\0" * 100)
    with pytest.raises(ValueError):
        lgsx_reader.open(bad)


def test_camera_projection():
    cam = lgsx_reader.Camera(
        id=1, waypoint_id=1, name="Rear Left", kind="pinhole", width=4096, height=3008, position=np.zeros(3), rotation=np.eye(3), fx=3600, fy=3600, cx=2048, cy=1504
    )
    uv = cam.project(np.array([[0, 0, -5.0], [1, 0.5, -5], [0, 0, 5]]))
    assert np.allclose(uv[0], [2048, 1504])
    assert uv[1, 0] > 2048 and uv[1, 1] < 1504
    assert np.isnan(uv[2]).all()


def test_las_writer(tmp_path):
    path = tmp_path / "out.las"
    with LasWriter(path, [100.0, 200.0, 0.0], [0.001] * 3, 'PROJCS["x"]') as w:
        w.write(np.array([[100.5, 200.25, 3.0], [101.0, 201.0, 4.0]]), rgb=np.array([[255, 0, 1], [2, 3, 4]], np.uint8), intensity=[7, 8], point_source_id=[3, 4])
    b = path.read_bytes()
    assert b[:4] == b"LASF" and b[24:26] == bytes([1, 4])
    (offset,) = struct.unpack_from("<I", b, 96)
    assert struct.unpack_from("<Q", b, 247)[0] == 2
    x, y, z = struct.unpack_from("<3i", b, offset)
    assert (x, y, z) == (500, 250, 3000)
    assert struct.unpack_from("<3H", b, offset + 30) == (65535, 0, 257)


def checksum(d):
    n = len(d["grid"])
    z = lambda k: d[k].astype(object) if k in d else np.zeros(n, dtype=object)  # noqa: E731
    g = d["grid"].astype(object)
    rgb = d["rgb"].astype(object) if "rgb" in d else np.zeros((n, 3), dtype=object)
    v = g[:, 0] + 3 * g[:, 1] + 7 * g[:, 2] + 11 * rgb[:, 0] + 13 * rgb[:, 1] + 17 * rgb[:, 2] + 19 * z("intensity") + 23 * z("classification") + 29 * z("setup") + 31 * z("normal_code")
    return sum((k + 1) * int(x) for k, x in enumerate(v))


@pytest.mark.parametrize(
    "path, members",
    [
        (MLS, [(0, 20867, 776993244658895), (275, 32651, 957614566307322), (1735, 262140, 132743320908013427)]),
        (TLS, [(1, 48962, 3885230749822038), (100, 71952, 7999730675259128), (206, 262137, 90708797298363229)]),
    ],
)
def test_real_files_match_the_reference(path, members):
    if not path or not os.path.exists(path):
        pytest.skip("the reference job is not given (set LGSX_MLS / LGSX_TLS to its .lgsx)")
    with lgsx_reader.open(path) as f:
        cloud = f.point_cloud()
        assert sum(n.point_count for n in cloud.nodes) == cloud.point_count
        for member, count, want in members:
            node = next(n for n in cloud.nodes if n.member == member)
            d = cloud.read_node(node)
            assert len(d["grid"]) == count
            assert checksum(d) == want, f"member {member}"
        p = f.project()
        assert p.point_count >= cloud.point_count
