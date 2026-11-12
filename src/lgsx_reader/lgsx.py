"""The LGSX container and its project.

An ``.lgsx`` ("Leica Project File X") starts with ``wild::geosystems`` and
is a flat container of named entries with a directory at the end: the last
8 bytes give the directory's offset; there, a u32 count and that many u64
offsets of the entry headers (u16 header length, i32 parent, u32 index, u16
length + name, u8 kind, u64 data offset, u64 size, u16 length + digest).

The project is a plain SQLite database (``ProjectDatabase``). Conventions,
checked against E57 exports of the same jobs: positions are world (or
project-frame) coordinates; quaternions are stored x, y, z, w; a setup's
rotation turns its frame into the project's; a camera's rotation turns
world into camera coordinates, and the camera looks along -z with +y up.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .points import PointCloud
from .props import parse_props

MAGIC = b"wild::geosystems"


@dataclass(frozen=True)
class Entry:
    name: str
    offset: int
    size: int
    kind: int


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """A rotation matrix from a quaternion (normalised first)."""
    q = np.array([w, x, y, z], float)
    n = np.linalg.norm(q)
    w, x, y, z = q / n if n > 0 else (1.0, 0.0, 0.0, 0.0)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class Setup:
    """A static scanner setup, or (mobile mapping) a track's frame."""

    id: int
    label: str
    setup_index: int
    resource_id: str
    position: np.ndarray
    rotation: np.ndarray  # 3x3, setup frame -> project frame


@dataclass
class Track:
    id: int
    name: str
    start_time: int  # unix seconds
    end_time: int
    setup_id: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Waypoint:
    id: int
    setup_id: int
    setup_index: int  # the points' setup index on a mobile-mapping job
    index: int  # along its track, from 0
    label: str
    time_ms: int  # from the track's start
    position: np.ndarray  # world
    rotation: np.ndarray  # 3x3, vehicle attitude (setup frame)


@dataclass
class Camera:
    """A mobile-mapping camera at a waypoint ("Rear Left", "Rear Right",
    "Sphere"). ``kind`` is "pinhole" or "equirectangular"."""

    id: int
    waypoint_id: int
    name: str
    kind: str
    width: int
    height: int
    position: np.ndarray  # projection centre, world
    rotation: np.ndarray  # 3x3, world -> camera (looks along -z, +y up)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)  # k1, k2, p1, p2, k3 (OpenCV order)
    entry: str = ""

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixel coordinates (n x 2: u right, v down, pixel centres at whole
        numbers) of world points; NaN behind a pinhole camera."""
        p = (np.atleast_2d(points) - self.position) @ self.rotation.T
        if self.kind == "pinhole":
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = p[:, 0] / -p[:, 2], p[:, 1] / -p[:, 2]
            k1, k2, p1, p2, k3 = self.distortion
            r2 = x * x + y * y
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            uv = np.stack([self.cx + self.fx * xd, self.cy - self.fy * yd], axis=1)
            uv[p[:, 2] >= 0] = np.nan
            return uv
        lon = np.arctan2(-p[:, 0], p[:, 2]) + np.pi
        lon = np.where(lon > np.pi, lon - 2 * np.pi, lon)
        lat = np.arcsin(p[:, 1] / np.linalg.norm(p, axis=1))
        return np.stack([(lon / (2 * np.pi) + 0.5) * self.width - 0.5, (0.5 - lat / np.pi) * self.height - 0.5], axis=1)


@dataclass
class Project:
    name: str | None
    point_count: int
    crs_name: str | None
    crs_wkt: str | None
    setups: list
    tracks: list
    waypoints: list
    cameras: list

    @property
    def is_mobile(self) -> bool:
        return bool(self.tracks)


#: Camera -> setup rotation of each ``CUBE`` face of a static setup's photos,
#: in blob order (quaternions x, y, z, w); checked against a Cyclone E57
#: export of the same faces.
_H = 2 ** -0.5
CUBE_FACE_ROTATIONS = [
    quat_to_matrix(_H, 0, 0, _H),
    quat_to_matrix(0, _H, _H, 0),
    quat_to_matrix(0.5, 0.5, 0.5, 0.5),
    quat_to_matrix(0.5, -0.5, -0.5, 0.5),
    quat_to_matrix(1, 0, 0, 0),
    quat_to_matrix(0, 0, 0, 1),
]


class Lgsx:
    """An open ``.lgsx`` file. Use as a context manager or call :meth:`close`."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self._f = open(self.path, "rb")
        try:
            self._f.seek(0, 2)
            self._len = self._f.tell()
            if self._len < 64 or self._read(0, 16) != MAGIC:
                raise ValueError(f"{self.path} is not a Leica LGSX file")
            self.entries = self._read_directory()
            self._by_name = {e.name: e for e in self.entries}
            self.info = self._project_info()
        except Exception:
            self._f.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._f.close()

    # ---- raw access ----------------------------------------------------------

    def _read(self, offset: int, size: int) -> bytes:
        self._f.seek(offset)
        b = self._f.read(size)
        if len(b) != size:
            raise ValueError(f"file cut short at {offset}")
        return b

    def _read_in(self, entry: Entry, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > entry.size:
            raise ValueError(f"read past the end of entry {entry.name!r}")
        return self._read(entry.offset + offset, size)

    def entry(self, name: str) -> Entry | None:
        return self._by_name.get(name)

    def read(self, name: str) -> bytes:
        """An entry's bytes by name ("ProjectDatabase", "SETUPCAMERA-12", ...)."""
        e = self._by_name.get(name)
        if e is None:
            raise KeyError(f"the LGSX has no {name!r} entry")
        return self._read(e.offset, e.size)

    def _read_directory(self) -> list:
        (d,) = struct.unpack("<Q", self._read(self._len - 8, 8))
        (count,) = struct.unpack("<I", self._read(d, 4))
        if count == 0 or d + 4 + 8 * count > self._len - 8:
            raise ValueError("the LGSX directory does not fit the file")
        offs = struct.unpack(f"<{count}Q", self._read(d + 4, 8 * count))
        lo, hi = min(offs), max(offs) + 2
        block = self._read(lo, min(max(d, hi + 512), self._len) - lo)
        out = []
        for o in offs:
            b = block[o - lo :]
            hlen, _, _, nl = struct.unpack_from("<HiIH", b, 0)
            p = 12 + nl
            name = b[12:p].rstrip(b"\0").decode("latin1")
            kind = b[p]
            off, size = struct.unpack_from("<QQ", b, p + 1)
            if off + size > self._len:
                raise ValueError(f"entry {name!r} runs past the end of the file")
            out.append(Entry(name, off, size, kind))
        return out

    def _project_info(self) -> dict:
        e = self._by_name.get("")
        if e is None:
            return {}
        b = self._read(e.offset, e.size)
        (nl,) = struct.unpack_from("<H", b, 2)
        return parse_props(b[4 + nl + 2 :])

    # ---- the project -----------------------------------------------------------

    def _database(self, name: str) -> sqlite3.Connection:
        data = self.read(name)
        if not data.startswith(b"SQLite format 3\0"):
            raise ValueError("the project database is not readable SQLite: the file is probably password-protected or encrypted")
        con = sqlite3.connect(":memory:")
        if hasattr(con, "deserialize"):
            con.deserialize(data)
            return con
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        tmp.write(data)
        tmp.close()
        return sqlite3.connect(tmp.name)

    def project(self) -> Project:
        """Setups, tracks, waypoints, cameras and the coordinate system."""
        db = self._database("ProjectDatabase")
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        for t in tables:
            cols = {r[1] for r in db.execute(f'pragma table_info("{t}")')}
            if "mPasswordType" in cols and db.execute(f'select count(*) from "{t}" where coalesce(mPasswordType, 0) != 0').fetchone()[0]:
                raise ValueError("the LGSX is password-protected")

        def rows(table, sql):
            return db.execute(sql).fetchall() if table in tables else []

        point_count = rows("Hds_OdbPointCloud", "select coalesce(sum(mPointCount), 0) from Hds_OdbPointCloud")
        crs = rows("Hds_OdbCoordSystem", "select mName, mWKT from Hds_OdbCoordSystem order by mId limit 1")
        setups = [
            Setup(r[0], r[1] or "", r[2] if r[2] is not None else -1, r[3] or "", np.array(r[4:7], float), quat_to_matrix(*r[7:11]))
            for r in rows(
                "Hds_OdbSetup",
                "select mId, mLabel, mSetupIndex, mResourceId, mPosition0, mPosition1, mPosition2, "
                "mRotation0, mRotation1, mRotation2, mRotation3 from Hds_OdbSetup order by mId",
            )
        ]
        by_setup = {s.id: s for s in setups}
        tracks = []
        for r in rows("Hds_OdbTrack", "select mId, mName, mStartTime, mEndTime, mpSetup, mMetadata from Hds_OdbTrack order by mId"):
            try:
                meta = parse_props(r[5] or b"")
            except ValueError:
                meta = {}
            tracks.append(Track(r[0], r[1] or "", r[2] or 0, r[3] or 0, r[4] if r[4] is not None else -1, meta))
        waypoints = []
        for r in rows(
            "Hds_OdbWayPoint",
            "select mId, mpRefSetup, mSetupIndex, mIndex, mLabel, mTimestamp, mPosition0, mPosition1, mPosition2, "
            "mRotation0, mRotation1, mRotation2, mRotation3 from Hds_OdbWayPoint order by mId",
        ):
            pos = np.array(r[6:9], float)
            s = by_setup.get(r[1])
            if s is not None:
                pos = s.position + s.rotation @ pos
            waypoints.append(Waypoint(r[0], r[1], r[2] if r[2] is not None else -1, r[3] or 0, r[4] or "", r[5] or 0, pos, quat_to_matrix(*r[9:13])))
        cameras = []
        for cid, wid, name, meta in rows("Hds_OdbSetupCamera", "select mId, mpWayPoint, mName, mMetadata from Hds_OdbSetupCamera order by mId"):
            p = parse_props(meta or b"")
            f = [p.get(f"Field_{i}", 0.0) for i in range(19)]
            common = dict(id=cid, waypoint_id=wid, name=name or "", width=p.get("ImgW", 0), height=p.get("ImgH", 0), entry=f"SETUPCAMERA-{cid}")
            if p.get("PropType") == 0:
                cameras.append(
                    Camera(kind="pinhole", position=np.array(f[12:15]), rotation=quat_to_matrix(*f[15:19]), cx=f[0], cy=f[1], fx=f[2], fy=f[3], distortion=tuple(f[4:9]), **common)
                )
            elif p.get("PropType") == 2:
                cameras.append(Camera(kind="equirectangular", position=np.array(f[6:9]), rotation=quat_to_matrix(*f[9:13]), **common))
        db.close()
        return Project(
            name=self.info.get("ProjectName"),
            point_count=int(point_count[0][0]) if point_count else 0,
            crs_name=crs[0][0] if crs else None,
            crs_wkt=crs[0][1] if crs else None,
            setups=setups,
            tracks=tracks,
            waypoints=waypoints,
            cameras=cameras,
        )

    def point_cloud(self) -> PointCloud:
        e = self._by_name.get("PointCloud")
        if e is None:
            raise KeyError("the LGSX has no point cloud")
        return PointCloud(self, e)

    # ---- photos and trajectory -------------------------------------------------

    def camera_image(self, camera: Camera) -> bytes:
        """A mobile-mapping photo (JPEG bytes)."""
        return self.read(camera.entry)

    def setup_cube(self, setup: Setup) -> list:
        """A static setup's colour photos: six 90-degree cube faces as
        ``(jpeg_bytes, rotation)``, rotation = camera -> project frame (the
        camera looks along -z, +y up; focal length (width - 1) / 2 pixels).
        Each JPEG is followed by its smaller mip levels, which decoders ignore."""
        blob = self.read(f"SETUP-camera-{setup.id}")
        if blob[:4] != b"CUBE" or len(blob) < 68:
            raise ValueError("not a CUBE image")
        starts = list(struct.unpack("<6Q", blob[-64:-16]))
        ends = starts[1:] + [len(blob) - 64]
        return [(blob[a:b], setup.rotation @ rot) for a, b, rot in zip(starts, ends, CUBE_FACE_ROTATIONS)]

    def trajectory(self, track_id: int) -> np.ndarray:
        """A track's waypoint poses (TMVF) as a structured array: ``time_ms``,
        ``xyz`` (relative to the track's setup) and ``q`` (x, y, z, w)."""
        b = self.read(f"TRACK-TrackMeasurementTrajectory-{track_id}")
        if b[:4] != b"TMVF":
            raise ValueError("not a TMVF trajectory")
        (count,) = struct.unpack_from("<I", b, 8)
        (rec,) = struct.unpack_from("<H", b, 16)
        dt = np.dtype({"names": ["time_ms", "xyz", "q"], "formats": ["<u8", ("<f8", 3), ("<f8", 4)], "offsets": [0, 8, 32], "itemsize": rec})
        return np.frombuffer(b, dt, count=count, offset=24)

