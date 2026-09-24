"""Camera geometry and pinhole projection utilities."""

import numpy as np
from numpy.typing import NDArray

from ..models import CameraPose


def create_image_pixel_grid(
    camera: CameraPose,
    image_resolution: int,
) -> tuple[
    NDArray[np.float64],
    int,
    int,
]:
    """Create a sampled pixel grid for a pinhole image.

    The image resolution specifies the number of samples along the
    longest image dimension. The original aspect ratio is preserved.

    Args:
        camera: Camera containing the image dimensions.
        image_resolution: Number of samples along the longest dimension.

    Returns:
        Sampled UV coordinates, sample width, and sample height.

    Raises:
        ValueError: If image_resolution is less than 2.
    """
    if image_resolution < 2:
        raise ValueError(
            "image_resolution must be at least 2."
        )

    scale = (
        image_resolution
        / max(
            camera.image_width,
            camera.image_height,
        )
    )

    sample_width = max(
        2,
        round(camera.image_width * scale),
    )

    sample_height = max(
        2,
        round(camera.image_height * scale),
    )

    u_values = np.linspace(
        0.0,
        camera.image_width - 1.0,
        sample_width,
        dtype=np.float64,
    )

    v_values = np.linspace(
        0.0,
        camera.image_height - 1.0,
        sample_height,
        dtype=np.float64,
    )

    u_grid, v_grid = np.meshgrid(
        u_values,
        v_values,
    )

    pixels_uv = np.column_stack(
        (
            u_grid.ravel(),
            v_grid.ravel(),
        )
    )

    return (
        pixels_uv,
        sample_width,
        sample_height,
    )


def pixels_to_world_plane(
    camera: CameraPose,
    pixels_uv: NDArray[np.float64],
    depth: float,
) -> NDArray[np.float64]:
    """Map image pixels onto a plane in world coordinates.

    This uses the same camera convention as the previously validated
    viewing-frustum geometry.

    Args:
        camera: Pinhole camera pose and intrinsics.
        pixels_uv: Image coordinates with shape (N, 2).
        depth: Image-plane distance from the camera centre in metres.

    Returns:
        World coordinates with shape (N, 3).

    Raises:
        ValueError: If depth is not positive.
    """
    if depth <= 0.0:
        raise ValueError(
            "depth must be greater than zero."
        )

    pixels_uv = np.asarray(
        pixels_uv,
        dtype=np.float64,
    )

    if (
        pixels_uv.ndim != 2
        or pixels_uv.shape[1] != 2
    ):
        raise ValueError(
            "pixels_uv must have shape (N, 2)."
        )

    u = pixels_uv[:, 0]
    v = pixels_uv[:, 1]

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
        pixels_uv.shape[0],
        -depth,
        dtype=np.float64,
    )

    camera_points = np.column_stack(
        (
            x,
            y,
            z,
        )
    )

    return transform_points(
        camera_points,
        camera.transform_matrix,
    )


def transform_points(
    points: NDArray[np.float64],
    transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform 3D points using a 4 x 4 transformation matrix."""
    homogeneous_points = np.column_stack(
        (
            points,
            np.ones(
                points.shape[0],
                dtype=np.float64,
            ),
        )
    )

    transformed = (
        transform
        @ homogeneous_points.T
    ).T

    return transformed[:, :3]
