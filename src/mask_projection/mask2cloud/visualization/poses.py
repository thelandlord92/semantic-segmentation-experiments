"""Functions for visualizing scanner and camera poses with Open3D."""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import open3d as o3d

from ..models import CameraPose, PointCloudData, ScannerPose


PoseGeometry: TypeAlias = (
    o3d.geometry.TriangleMesh
    | o3d.geometry.LineSet
)


def create_pose_geometries(
    scanner_poses: Sequence[ScannerPose],
    point_clouds: Sequence[PointCloudData] | None = None,
    marker_radius: float = 0.15,
    scanner_frame_size: float = 0.75,
    camera_frame_size: float = 0.35,
    frustum_depth: float = 1.0,
    show_scanner_frames: bool = True,
    show_camera_frames: bool = False,
    show_camera_frustums: bool = True,
) -> list[PoseGeometry]:
    """Create scanner and camera visualization geometries.

    Args:
        scanner_poses: Scanner poses to visualize.
        point_clouds: Optional loaded point clouds. When supplied, only
            poses associated with loaded E57 files are shown.
        marker_radius: Radius of scanner-centre markers in metres.
        scanner_frame_size: Size of scanner coordinate frames.
        camera_frame_size: Size of camera coordinate frames.
        frustum_depth: Distance from camera centre to the displayed
            image plane in metres.
        show_scanner_frames: Whether to display scanner coordinate frames.
        show_camera_frames: Whether to display camera coordinate frames.
        show_camera_frustums: Whether to display camera viewing frustums.

    Returns:
        Open3D geometries representing the scanner and camera poses.
    """
    poses = _filter_scanner_poses(
        scanner_poses,
        point_clouds,
    )

    geometries: list[PoseGeometry] = []

    for scanner_pose in poses:
        geometries.append(
            _create_scanner_marker(
                scanner_pose,
                radius=marker_radius,
            )
        )

        if show_scanner_frames:
            geometries.append(
                _create_scanner_coordinate_frame(
                    scanner_pose,
                    size=scanner_frame_size,
                )
            )

        if show_camera_frames:
            geometries.extend(
                create_camera_pose_geometries(
                    scanner_pose,
                    frame_size=camera_frame_size,
                )
            )

        if show_camera_frustums:
            geometries.extend(
                create_camera_frustum_geometries(
                    scanner_pose,
                    depth=frustum_depth,
                )
            )

    return geometries


def create_camera_pose_geometries(
    scanner_pose: ScannerPose,
    frame_size: float = 0.35,
) -> list[o3d.geometry.TriangleMesh]:
    """Create coordinate frames for the cameras of one scanner pose."""
    geometries = []

    for camera in scanner_pose.cameras:
        coordinate_frame = (
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size,
            )
        )

        coordinate_frame.transform(
            camera.transform_matrix
        )

        geometries.append(
            coordinate_frame
        )

    return geometries


def create_camera_frustum_geometries(
    scanner_pose: ScannerPose,
    depth: float = 1.0,
) -> list[o3d.geometry.LineSet]:
    """Create viewing frustums for all cameras at one scanner pose.

    Args:
        scanner_pose: Scanner pose containing the camera poses.
        depth: Visualization depth of the image plane in metres.

    Returns:
        One Open3D LineSet for each pinhole camera.
    """
    return [
        create_camera_frustum_geometry(
            camera,
            depth=depth,
        )
        for camera in scanner_pose.cameras
    ]


def create_camera_frustum_geometry(
    camera: CameraPose,
    depth: float = 1.0,
    color: tuple[float, float, float] = (
        1.0,
        0.75,
        0.0,
    ),
) -> o3d.geometry.LineSet:
    """Create a pinhole-camera viewing frustum.

    The frustum is first constructed in the local E57 camera
    coordinate system and then transformed into world coordinates.

    Args:
        camera: Pinhole camera pose and intrinsics.
        depth: Distance of the displayed image plane from the
            camera centre in metres.
        color: RGB color of the frustum lines.

    Returns:
        Open3D LineSet representing the viewing frustum.

    Raises:
        ValueError: If depth is not positive.
    """
    if depth <= 0.0:
        raise ValueError(
            "Frustum depth must be greater than zero."
        )

    corners_local = _calculate_image_plane_corners(
        camera,
        depth=depth,
    )

    camera_origin = np.zeros(
        3,
        dtype=np.float64,
    )

    optical_axis_point = np.array(
        [0.0, 0.0, -depth],
        dtype=np.float64,
    )

    points_local = np.vstack(
        [
            camera_origin,
            corners_local,
            optical_axis_point,
        ]
    )

    points_world = _transform_points(
        points_local,
        camera.transform_matrix,
    )

    lines = np.array(
        [
            # Rays from camera centre to image corners.
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],

            # Image-plane rectangle.
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],

            # Optical axis.
            [0, 5],
        ],
        dtype=np.int32,
    )

    frustum = o3d.geometry.LineSet()

    frustum.points = o3d.utility.Vector3dVector(
        points_world
    )

    frustum.lines = o3d.utility.Vector2iVector(
        lines
    )

    frustum.paint_uniform_color(
        color
    )

    return frustum


def _calculate_image_plane_corners(
    camera: CameraPose,
    depth: float,
) -> np.ndarray:
    """Calculate image-plane corners in camera coordinates.

    E57 pinhole cameras view along the negative local Z axis.

    Args:
        camera: Camera pose containing pinhole intrinsics.
        depth: Distance of the image plane from the camera centre.

    Returns:
        Four image-plane corners with shape (4, 3).
    """
    u_min = -0.5
    u_max = camera.image_width - 0.5

    v_min = -0.5
    v_max = camera.image_height - 0.5

    image_corners = np.array(
        [
            [u_min, v_min],
            [u_max, v_min],
            [u_max, v_max],
            [u_min, v_max],
        ],
        dtype=np.float64,
    )

    u = image_corners[:, 0]
    v = image_corners[:, 1]

    x = (
        (u - camera.cx)
        / camera.fx
        * depth
    )

    y = (
        (v - camera.cy)
        / camera.fy
        * depth
    )

    z = np.full(
        4,
        -depth,
        dtype=np.float64,
    )

    return np.column_stack(
        [
            x,
            y,
            z,
        ]
    )


def _transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Transform 3D points using a 4 x 4 transformation matrix."""
    homogeneous_points = np.column_stack(
        [
            points,
            np.ones(
                points.shape[0],
                dtype=np.float64,
            ),
        ]
    )

    transformed = (
        transform
        @ homogeneous_points.T
    ).T

    return transformed[:, :3]


def _create_scanner_marker(
    scanner_pose: ScannerPose,
    radius: float,
) -> o3d.geometry.TriangleMesh:
    """Create a sphere at the scanner centre."""
    marker = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
    )

    marker.compute_vertex_normals()

    marker.translate(
        scanner_pose.scanner_center_world
    )

    return marker


def _create_scanner_coordinate_frame(
    scanner_pose: ScannerPose,
    size: float,
) -> o3d.geometry.TriangleMesh:
    """Create an oriented coordinate frame for a scanner pose."""
    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size,
        )
    )

    coordinate_frame.transform(
        scanner_pose.transform_matrix
    )

    return coordinate_frame


def _filter_scanner_poses(
    scanner_poses: Sequence[ScannerPose],
    point_clouds: Sequence[PointCloudData] | None,
) -> list[ScannerPose]:
    """Filter poses to scanner files present in the scene."""
    poses = list(scanner_poses)

    if point_clouds is None:
        return poses

    loaded_files = {
        cloud.source_path.name
        for cloud in point_clouds
    }

    return [
        pose
        for pose in poses
        if pose.source_e57_file in loaded_files
    ]