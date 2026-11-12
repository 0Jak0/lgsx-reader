"""``lgsx``: look inside and convert Leica LGSX files.

    lgsx info JOB.lgsx
    lgsx to-las JOB.lgsx OUT.las [--every N]
    lgsx waypoints JOB.lgsx OUT.csv
    lgsx images JOB.lgsx OUTDIR [--camera Sphere] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import numpy as np

from . import __version__
from .las import LasWriter
from .lgsx import Lgsx


def _safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*#]+', "_", name).strip() or "unnamed"


def cmd_info(a) -> int:
    with Lgsx(a.file) as f:
        p = f.project()
        cloud = f.point_cloud()
        h = cloud.header
        print(f"{a.file}")
        print(f"  project      {p.name}  (LGS version {f.info.get('LGSVersion')})")
        print(f"  coordinates  {p.crs_name or 'local frame (no coordinate system)'}")
        print(f"  points       {h.point_count:,} in {len(cloud.nodes):,} nodes" + (f" (+{h.filtered_point_count:,} filtered out)" if h.filtered_point_count else ""))
        print(f"  attributes   {', '.join(f'{n}:{t}' for n, t in h.attributes)}")
        print(f"  bounds       {np.round(h.bounds_min, 3).tolist()} .. {np.round(h.bounds_max, 3).tolist()}")
        if p.is_mobile:
            print(f"  mobile       {len(p.tracks)} tracks, {len(p.waypoints):,} waypoints, {len(p.cameras):,} photos")
            for t in p.tracks:
                n = sum(1 for w in p.waypoints if w.setup_id == t.setup_id)
                m = t.metadata
                print(f"    {t.name}: {n} waypoints, {t.end_time - t.start_time} s, {m.get('ScannerModel', '')} {m.get('ScannerSerialNumber', '')}".rstrip())
            names = sorted({c.name for c in p.cameras})
            print(f"  cameras      {', '.join(names)}")
        else:
            print(f"  static       {len(p.setups)} setups")
            for s in p.setups:
                photos = "photos" if f.entry(f"SETUP-camera-{s.id}") else "no photos"
                print(f"    {s.label}: at {np.round(s.position, 3).tolist()}, {photos}")
        size = sum(e.size for e in f.entries if e.name.startswith(("SETUPCAMERA-", "SETUP-")))
        print(f"  images       {size / 1e9:.2f} GB")
    return 0


def cmd_to_las(a) -> int:
    t = time.time()
    with Lgsx(a.file) as f:
        p = f.project()
        cloud = f.point_cloud()
        h = cloud.header
        written = 0
        with LasWriter(a.out, h.origin, h.scale, p.crs_wkt) as w:
            for k, (node, d) in enumerate(cloud.iter_nodes()):
                sel = slice(None, None, a.every)
                w.write(
                    d["xyz"][sel],
                    rgb=d["rgb"][sel] if "rgb" in d else None,
                    intensity=d["intensity"][sel] if "intensity" in d else None,
                    classification=d["classification"][sel] if "classification" in d else None,
                    point_source_id=d["setup"][sel] if "setup" in d else None,
                )
                written += len(d["xyz"][sel])
                if k % 50 == 0:
                    print(f"\r  {written:,} of about {h.point_count // a.every:,} points", end="", file=sys.stderr)
    print(f"\r{a.out}: {written:,} points in {time.time() - t:.0f} s" + " " * 20)
    if "setup" in d:
        print("  point source ID = the setup index (lgsx waypoints / info: its label)")
    return 0


def cmd_waypoints(a) -> int:
    with Lgsx(a.file) as f:
        p = f.project()
        with open(a.out, "w", newline="") as o:
            out = csv.writer(o)
            out.writerow(["setup_index", "label", "setup", "time_ms", "x", "y", "z", "photos"])
            if p.is_mobile:
                shots = {}
                for c in p.cameras:
                    shots.setdefault(c.waypoint_id, []).append(c.name)
                for wp in p.waypoints:
                    out.writerow([wp.setup_index, wp.label, wp.setup_id, wp.time_ms, *np.round(wp.position, 4), "; ".join(shots.get(wp.id, []))])
            else:
                for s in p.setups:
                    out.writerow([s.setup_index, s.label, s.id, "", *np.round(s.position, 4), "cube" if f.entry(f"SETUP-camera-{s.id}") else ""])
    print(a.out)
    return 0


def cmd_images(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with Lgsx(a.file) as f:
        p = f.project()
        if p.is_mobile:
            labels = {w.id: w.label for w in p.waypoints}
            for c in p.cameras:
                if a.camera and c.name != a.camera:
                    continue
                path = os.path.join(a.out, _safe(f"{labels.get(c.waypoint_id, c.waypoint_id)} {c.name}") + ".jpg")
                with open(path, "wb") as o:
                    o.write(f.camera_image(c))
                n += 1
                if a.limit and n >= a.limit:
                    break
        else:
            for s in p.setups:
                if not f.entry(f"SETUP-camera-{s.id}"):
                    continue
                for k, (jpeg, _) in enumerate(f.setup_cube(s)):
                    with open(os.path.join(a.out, _safe(f"{s.label} face{k}") + ".jpg"), "wb") as o:
                        o.write(jpeg)
                    n += 1
                if a.limit and n >= a.limit:
                    break
    print(f"{n} photos in {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lgsx", description="Look inside and convert Leica LGSX (TruView) files.")
    ap.add_argument("--version", action="version", version=f"lgsx-reader {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="what the file holds")
    s.add_argument("file")
    s.set_defaults(fn=cmd_info)
    s = sub.add_parser("to-las", help="the points as LAS 1.4 (colour, intensity, class, setup as point source ID, CRS)")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("--every", type=int, default=1, help="keep every Nth point (default 1: all)")
    s.set_defaults(fn=cmd_to_las)
    s = sub.add_parser("waypoints", help="waypoints (mobile) or setups (static) as CSV")
    s.add_argument("file")
    s.add_argument("out")
    s.set_defaults(fn=cmd_waypoints)
    s = sub.add_parser("images", help="extract the photos as JPEG")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("--camera", help='only this camera, e.g. "Sphere", "Rear Left" (mobile mapping)')
    s.add_argument("--limit", type=int, default=0, help="stop after N photos")
    s.set_defaults(fn=cmd_images)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (OSError, ValueError, KeyError) as e:
        print(f"lgsx: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
