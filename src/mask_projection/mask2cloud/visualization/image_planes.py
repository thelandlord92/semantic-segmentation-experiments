"""Functions for visualizing pinhole images in 3D."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open3d as o3d

from ..geometry.camera import (
    create_image_pixel_grid,
    pixels_to_world_plane,
)
from ..models import (
    CameraPose,
    PointCloudData,
    ScannerPose,
)


def create_camera_image_plane_geometries(
    scanner_poses: Sequence[ScannerPose],
    image_dir: str | Path,
    point_clouds: Sequence[PointCloudData] | None = None,
    depth: float = 1.0,
    image_resolution: int = 64,
) -> list[o3d.geometry.TriangleMesh]:
    """Create coloured image planes for pinhole cameras.

    Args:
        scanner_poses: Scanner poses containing camera information.
        image_dir: Directory containing the pinhole JPEG images.
        point_clouds: Optional point clouds used to filter scanner poses.
        depth: Distance of the image planes from their camera centres.
        image_resolution: Samples along the longest image dimension.

    Returns:
        Coloured Open3D image-plane meshes.
    """
    image_path = Path(image_dir).expanduser().resolve()

    if not image_path.is_dir():
        raise NotADirectoryError(
            f"Image directory not found: {image_path}"
        )

    poses = _filter_scanner_poses(
        scanner_poses,
        point_clouds,
    )

    geometries = []

    for scanner_pose in poses:
        for camera in scanner_pose.cameras:
            geometry = create_camera_image_plane_geometry(
                camera,
                image_dir=image_path,
                depth=depth,
                image_resolution=image_resolution,
            )

            geometries.append(
                geometry
            )

    return geometries


def create_camera_image_plane_geometry(
    camera: CameraPose,
    image_dir: str | Path,
    depth: float = 1.0,
    image_resolution: int = 64,
) -> o3d.geometry.TriangleMesh:
    """Create a coloured 3D image plane for one camera.

    Args:
        camera: Camera pose and pinhole intrinsics.
        image_dir: Directory containing the camera image.
        depth: Distance of the image plane from the camera centre.
        image_resolution: Samples along the longest image dimension.

    Returns:
        Open3D triangle mesh representing the image plane.

    Raises:
        FileNotFoundError: If the camera image cannot be found.
    """
    image_path = (
        Path(image_dir)
        / camera.image_file
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Camera image not found: {image_path}"
        )

    pixels_uv, sample_width, sample_height = (
        create_image_pixel_grid(
            camera,
            image_resolution=image_resolution,
        )
    )

    vertices = pixels_to_world_plane(
        camera,
        pixels_uv,
        depth=depth,
    )

    colors = _sample_image_colors(
        image_path,
        pixels_uv,
    )

    triangles = _create_grid_triangles(
        sample_width,
        sample_height,
    )

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(
        vertices
    )

    mesh.triangles = o3d.utility.Vector3iVector(
        triangles
    )

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        colors
    )

    mesh.compute_vertex_normals()

    return mesh


def _sample_image_colors(
    image_path: Path,
    pixels_uv: np.ndarray,
) -> np.ndarray:
    """Sample RGB values from an image at UV coordinates."""
    image = np.asarray(
        o3d.io.read_image(
            str(image_path)
        )
    )

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"Expected RGB image: {image_path}"
        )

    u_indices = np.rint(
        pixels_uv[:, 0]
    ).astype(np.int64)

    v_indices = np.rint(
        pixels_uv[:, 1]
    ).astype(np.int64)

    u_indices = np.clip(
        u_indices,
        0,
        image.shape[1] - 1,
    )

    v_indices = np.clip(
        v_indices,
        0,
        image.shape[0] - 1,
    )

    colors = image[
        v_indices,
        u_indices,
        :3,
    ].astype(np.float64)

    if colors.max() > 1.0:
        colors /= 255.0

    return colors


def _create_grid_triangles(
    width: int,
    height: int,
) -> np.ndarray:
    """Create triangle indices for a regular image grid."""
    rows = np.arange(
        height - 1,
        dtype=np.int64,
    )

    columns = np.arange(
        width - 1,
        dtype=np.int64,
    )

    column_grid, row_grid = np.meshgrid(
        columns,
        rows,
    )

    top_left = (
        row_grid.ravel() * width
        + column_grid.ravel()
    )

    top_right = top_left + 1
    bottom_left = top_left + width
    bottom_right = bottom_left + 1

    first_triangles = np.column_stack(
        (
            top_left,
            bottom_left,
            top_right,
        )
    )

    second_triangles = np.column_stack(
        (
            top_right,
            bottom_left,
            bottom_right,
        )
    )

    return np.vstack(
        (
            first_triangles,
            second_triangles,
        )
    ).astype(np.int32)


def _filter_scanner_poses(
    scanner_poses: Sequence[ScannerPose],
    point_clouds: Sequence[PointCloudData] | None,
) -> list[ScannerPose]:
    """Filter poses to E57 files represented by loaded clouds."""
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