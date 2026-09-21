"""Functions for visualizing point-cloud data with Open3D."""

from collections.abc import Sequence

import numpy as np
import open3d as o3d

from ..models import PointCloudData


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
        colors = _normalize_colors(point_cloud.colors)

        geometry.colors = o3d.utility.Vector3dVector(
            colors
        )

    return geometry


def visualize_point_cloud(
    point_cloud: PointCloudData | Sequence[PointCloudData],
    use_colors: bool = True,
    point_size: float = 1.0,
    show_coordinate_frame: bool = True,
    coordinate_frame_size: float = 1.0,
    window_name: str = "mask2cloud",
    width: int = 1600,
    height: int = 900,
) -> None:
    """Visualize one or more point clouds with Open3D.

    Args:
        point_cloud: One point cloud or a sequence of point clouds.
        use_colors: Whether to display RGB values when available.
        point_size: Size of points in the Open3D viewer.
        show_coordinate_frame: Whether to show the world XYZ frame.
        coordinate_frame_size: Size of the world coordinate frame.
        window_name: Name of the visualization window.
        width: Width of the visualization window in pixels.
        height: Height of the visualization window in pixels.
    """
    if isinstance(point_cloud, PointCloudData):
        point_clouds = [point_cloud]
    else:
        point_clouds = list(point_cloud)

    if not point_clouds:
        raise ValueError("At least one point cloud is required.")

    geometries = [
        to_open3d_point_cloud(
            cloud,
            use_colors=use_colors,
        )
        for cloud in point_clouds
    ]

    if show_coordinate_frame:
        coordinate_frame = (
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=coordinate_frame_size,
                origin=[0.0, 0.0, 0.0],
            )
        )
        geometries.append(coordinate_frame)

    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        window_name=window_name,
        width=width,
        height=height,
    )

    for geometry in geometries:
        visualizer.add_geometry(geometry)

    render_options = visualizer.get_render_option()

    if render_options is not None:
        render_options.point_size = point_size

    visualizer.run()
    visualizer.destroy_window()


def _normalize_colors(
    colors: np.ndarray,
) -> np.ndarray:
    """Normalize RGB values to the Open3D range of 0 to 1.

    Args:
        colors: RGB values with shape ``(N, 3)``.

    Returns:
        RGB values as floating-point values between 0 and 1.

    Raises:
        ValueError: If the color array does not have shape ``(N, 3)``.
    """
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