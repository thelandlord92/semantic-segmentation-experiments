"""Functions for visualizing scanner poses with Open3D."""

from collections.abc import Sequence

import numpy as np
import open3d as o3d

from ..models import PointCloudData, ScannerPose


def create_scanner_pose_geometries(
    scanner_poses: Sequence[ScannerPose],
    point_clouds: Sequence[PointCloudData] | None = None,
    marker_radius: float = 0.15,
    coordinate_frame_size: float = 0.75,
) -> list:
    """Create Open3D geometries representing scanner poses.

    Args:
        scanner_poses: Scanner poses to visualize.
        point_clouds: Optional loaded point clouds. When supplied, only
            scanner poses associated with loaded E57 files are shown.
        marker_radius: Radius of scanner-centre markers in metres.
        coordinate_frame_size: Length of coordinate-frame axes in metres.

    Returns:
        Open3D geometries representing scanner centres and orientations.
    """
    poses = list(scanner_poses)

    if point_clouds is not None:
        loaded_files = {
            cloud.source_path.name
            for cloud in point_clouds
        }

        poses = [
            pose
            for pose in poses
            if pose.source_e57_file in loaded_files
        ]

    geometries = []

    for pose in poses:
        marker = _create_scanner_marker(
            pose,
            radius=marker_radius,
        )

        coordinate_frame = _create_scanner_coordinate_frame(
            pose,
            size=coordinate_frame_size,
        )

        geometries.extend(
            [
                marker,
                coordinate_frame,
            ]
        )

    return geometries


def _create_scanner_marker(
    pose: ScannerPose,
    radius: float,
) -> o3d.geometry.TriangleMesh:
    """Create a sphere at the scanner centre."""
    marker = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
    )

    marker.compute_vertex_normals()

    marker.translate(
        pose.scanner_center_world
    )

    return marker


def _create_scanner_coordinate_frame(
    pose: ScannerPose,
    size: float,
) -> o3d.geometry.TriangleMesh:
    """Create an oriented coordinate frame for a scanner pose."""
    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size,
        )
    )

    transform = np.asarray(
        pose.transform_matrix,
        dtype=np.float64,
    )

    coordinate_frame.transform(transform)

    return coordinate_frame