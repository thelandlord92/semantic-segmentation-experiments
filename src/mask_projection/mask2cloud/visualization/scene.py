"""Functions for displaying Open3D scenes."""

from collections.abc import Sequence

import open3d as o3d


def visualize_scene(
    geometries: Sequence,
    point_size: float = 1.0,
    show_world_frame: bool = True,
    world_frame_size: float = 1.0,
    window_name: str = "mask2cloud",
    width: int = 1600,
    height: int = 900,
) -> None:
    """Display Open3D geometries in a single visualization window.

    Args:
        geometries: Open3D geometries to display.
        point_size: Point size used for point-cloud geometries.
        show_world_frame: Whether to display the world coordinate frame.
        world_frame_size: Size of the world coordinate frame.
        window_name: Visualization window title.
        width: Window width in pixels.
        height: Window height in pixels.

    Raises:
        ValueError: If no geometries are provided.
    """
    scene_geometries = list(geometries)

    if not scene_geometries:
        raise ValueError(
            "At least one geometry is required."
        )

    if show_world_frame:
        world_frame = (
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=world_frame_size,
                origin=[0.0, 0.0, 0.0],
            )
        )

        scene_geometries.append(world_frame)

    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        window_name=window_name,
        width=width,
        height=height,
    )

    for geometry in scene_geometries:
        visualizer.add_geometry(geometry)

    render_options = visualizer.get_render_option()

    if render_options is not None:
        render_options.point_size = point_size

    visualizer.run()
    visualizer.destroy_window()