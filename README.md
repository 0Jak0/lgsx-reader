# lgsx-reader

Read Leica **LGSX** files, the single-file projects Leica TruView opens, as
published by Pegasus OFFICE (mobile mapping) and Cyclone (static scans): the
project, the point cloud, the trajectory and the photos. Convert the points to
LAS. Python 3.9+, needs only numpy and zstandard.

Independent work, not affiliated with or endorsed by Leica Geosystems or
Hexagon. It reads files; it does not write them.

## Install

```bash
pip install .
```

## Command line

```bash
lgsx info JOB.lgsx
lgsx to-las JOB.lgsx OUT.las
lgsx waypoints JOB.lgsx OUT.csv
lgsx images JOB.lgsx OUTDIR --camera Sphere
```

| Command | What it does |
|---|---|
| `info` | Project name, coordinate system, point count and attributes, tracks and waypoints or setups, cameras, image size |
| `to-las` | Every point as LAS 1.4 (point format 7): position on the file's own grid (exact), 16-bit RGB, intensity, class, and the setup / waypoint as point source ID; the coordinate system as WKT. `--every N` thins |
| `waypoints` | Mobile mapping: each waypoint's setup index, label, time and position, and its cameras. Static: each setup |
| `images` | The photos as JPEG: every waypoint's cameras (`--camera "Rear Left"` for one), or each setup's six cube faces. `--limit N` |

## Python

```python
import lgsx_reader

with lgsx_reader.open("job.lgsx") as f:
    project = f.project()            # setups, tracks, waypoints, cameras, CRS
    cloud = f.point_cloud()          # header + nodes
    for node, pts in cloud.iter_nodes():
        xyz = pts["xyz"]             # n x 3 float64, world / project coordinates
        rgb = pts.get("rgb")         # n x 3 uint8
        setup = pts.get("setup")     # index into cloud.header.setup_labels
    normals = lgsx_reader.normals(pts["normal_code"])   # static scans

    cam = project.cameras[0]         # mobile mapping
    jpeg = f.camera_image(cam)
    uv = cam.project(xyz)            # where points land in the photo (pixels)
    poses = f.trajectory(project.tracks[0].id)
```

Conventions: positions are world (or project-frame) coordinates; rotation
matrices are built from quaternions stored x, y, z, w; a setup's rotation turns
its frame into the project's; a camera's turns world into camera coordinates,
and the camera looks along -z with +y up. Rear (pinhole) cameras use
Brown-Conrady distortion k1, k2, p1, p2, k3; the pano is equirectangular with its
centre column along -z.

## The format, briefly

* **Container:** `wild::geosystems`, named entries, a directory at the end.
* **Project:** a plain SQLite database (`Hds_Odb*` tables).
* **Points:** a Hexagon HSPC archive. Octree nodes hold each attribute as
  zstd-compressed byte planes. Every plane is pair-swapped. Positions are
  delta-coded 96-bit Morton codes on a 1 mm grid; other attributes are
  delta-coded U8/U16, and colour is G plus offsets for B and R. Normals are
  16-bit codes on a 226-ring table.
* **Photos:** JPEG per camera (mobile); `CUBE` blobs of six faces with mip
  levels (static).
* **Trajectory:** `TMVF`, one pose per waypoint.

## How it was checked

* **A Pegasus mobile-mapping job (4 tracks, 178.6 M points):** checked against
  Pegasus OFFICE's E57 export of the same job, on 50 random nodes (4.3 M
  points). Position is within 1 mm, class matches 100%, and RGB, intensity and
  waypoint 99.99%. The pano and rear photos line up with the cloud.
* **A Cyclone static job (12 setups, 25.1 M points):** `lgsx to-las` matches
  Cyclone's own LAS export point for point. Position, RGB and intensity agree
  for all 25,060,315 points. Normals face their scanner for 100% of points.
* The tests (`pytest tests`) check every part of the format on synthetic files.
  They also check node checksums on those two jobs, when given their paths in
  `LGSX_MLS` and `LGSX_TLS`; these agree with a
  separate Rust reader.

## Limits

* Password-protected or encrypted files are refused.
* The entries' digests are not verified (they look like Leica signatures),
  and no file is written.
* HSPC format version 22 only; other versions are refused with a message.
* The normal decoding is fitted to normals estimated from the points (median
  error 3 degrees, the noise of that estimate).
* The rear cameras' distortion order was chosen by how well the photos line
  up with the cloud.

## Licence

MIT (see `LICENSE`): use, copy, change and pass it on freely, keeping the
copyright notice.
