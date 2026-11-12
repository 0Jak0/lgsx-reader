"""Read Leica LGSX files (the single-file projects Leica TruView opens, as
published by Pegasus OFFICE and Cyclone): the project, the point cloud, the
trajectory and the photos.

    import lgsx_reader

    with lgsx_reader.open("job.lgsx") as f:
        project = f.project()
        cloud = f.point_cloud()
        for node, pts in cloud.iter_nodes():
            pts["xyz"], pts["rgb"], pts["intensity"]  # numpy arrays

Independent work, not affiliated with or endorsed by Leica Geosystems or
Hexagon. Password-protected files are refused.
"""

from .las import LasWriter
from .lgsx import CUBE_FACE_ROTATIONS, Camera, Entry, Lgsx, Project, Setup, Track, Waypoint, quat_to_matrix
from .points import Header, Node, PointCloud, decode_node, normals
from .props import parse_props

__version__ = "0.1.0"

__all__ = [
    "CUBE_FACE_ROTATIONS",
    "Camera",
    "Entry",
    "Header",
    "LasWriter",
    "Lgsx",
    "Node",
    "PointCloud",
    "Project",
    "Setup",
    "Track",
    "Waypoint",
    "decode_node",
    "normals",
    "open",
    "parse_props",
    "quat_to_matrix",
]


def open(path) -> Lgsx:  # noqa: A001 - like io.open
    """Open an ``.lgsx`` file (use as a context manager)."""
    return Lgsx(path)
