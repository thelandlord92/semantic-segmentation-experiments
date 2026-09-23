"""Functions for creating Open3D point-cloud geometries."""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import open3d as o3d

from ..models import PointCloudData


PointCloudInput: TypeAlias = (
    PointCloudData | Sequence[PointCloudData]
)


def create_point_cloud_geometries(
    point_clouds: PointCloudInput,
    use_colors: bool = True,
) -> list[o3d.geometry.PointCloud]:
    """Create Open3D geometries from point-cloud data.

    Args:
        point_clouds: One PointCloudData object or a sequence of
            PointCloudData objects.
        use_colors: Whether to include RGB values when available.

    Returns:
        Open3D point-cloud geometries.
    """
    clouds = _normalize_point_cloud_input(point_clouds)

    return [
        to_open3d_point_cloud(
            cloud,
            use_colors=use_colors,
        )
        for cloud in clouds
    ]


def to_open3d_point_cloud(
    point_cloud: PointCloudData,
    use_colors: bool = True,
) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to an Open3D point cloud.

    Args:
        point_cloud: Point-cloud data to convert.
        use_colors: Whether to include RGB values when available.

    Returns:
        Open3D point-cloud geometry.
    """
    geometry = o3d.geometry.PointCloud()

    geometry.points = o3d.utility.Vector3dVector(
        point_cloud.points
    )

    if use_colors and point_cloud.colors is not None:
        geometry.colors = o3d.utility.Vector3dVector(
            _normalize_colors(point_cloud.colors)
        )

    return geometry


def _normalize_point_cloud_input(
    point_clouds: PointCloudInput,
) -> list[PointCloudData]:
    """Normalize point-cloud input to a list.

    Args:
        point_clouds: One PointCloudData object or a sequence of them.

    Returns:
        List of PointCloudData objects.

    Raises:
        ValueError: If no point clouds are supplied.
        TypeError: If an unsupported object is supplied.
    """
    if isinstance(point_clouds, PointCloudData):
        return [point_clouds]

    clouds = list(point_clouds)

    if not clouds:
        raise ValueError(
            "At least one point cloud is required."
        )

    if not all(
        isinstance(cloud, PointCloudData)
        for cloud in clouds
    ):
        raise TypeError(
            "All items must be PointCloudData objects."
        )

    return clouds


def _normalize_colors(
    colors: np.ndarray,
) -> np.ndarray:
    """Normalize RGB values to the Open3D range of 0 to 1."""
    colors = np.asarray(
        colors,
        dtype=np.float64,
    )

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(
            "Colors must have shape (N, 3)."
        )

    maximum = np.nanmax(colors)

    if maximum <= 1.0:
        return np.clip(colors, 0.0, 1.0)

    if maximum <= 255.0:
        colors = colors / 255.0
    elif maximum <= 65535.0:
        colors = colors / 65535.0
    else:
        colors = colors / maximum

    return np.clip(colors, 0.0, 1.0)
