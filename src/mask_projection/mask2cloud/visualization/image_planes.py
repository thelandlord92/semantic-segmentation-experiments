"""Functions for visualizing pinhole images and segmentations in 3D."""

import colorsys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d

from ..geometry.camera import (
    create_image_pixel_grid,
    pixels_to_world_plane,
)
from ..geometry.image_mapping import (
    extract_camera_image_key,
    orient_camera_array,
)
from ..geometry.segmentation import (
    decode_coco_segmentation,
    find_mask_boundary,
    sample_mask,
)
from ..models import (
    CameraPose,
    CocoSegmentationData,
    PointCloudData,
    ScannerPose,
)


def create_camera_image_plane_geometries(
    scanner_poses: Sequence[ScannerPose],
    image_dir: str | Path,
    point_clouds: Sequence[PointCloudData] | None = None,
    segmentation_data: CocoSegmentationData | None = None,
    depth: float = 1.0,
    image_resolution: int = 64,
    image_mode: Literal["color", "grayscale"] = "color",
    segmentation_opacity: float = 0.65,
    only_segmented_images: bool = True,
) -> list[o3d.geometry.TriangleMesh]:
    """Create image-plane geometries for pinhole cameras.

    Args:
        scanner_poses: Scanner poses containing camera information.
        image_dir: Directory containing the pinhole images.
        point_clouds: Optional point clouds used to filter scanner poses.
        segmentation_data: Optional filtered COCO segmentation data.
        depth: Distance of the image planes from their camera centres.
        image_resolution: Samples along the longest image dimension.
        image_mode: Display source images in color or grayscale.
        segmentation_opacity: Opacity of segmentation overlays.
        only_segmented_images: If True and segmentation data are supplied,
            display only images containing selected segmentations.

    Returns:
        Open3D triangle meshes representing camera image planes.
    """
    image_path = Path(image_dir).expanduser().resolve()

    if not image_path.is_dir():
        raise NotADirectoryError(
            f"Image directory not found: {image_path}"
        )

    if not 0.0 <= segmentation_opacity <= 1.0:
        raise ValueError(
            "segmentation_opacity must be between 0 and 1."
        )

    poses = _filter_scanner_poses(
        scanner_poses,
        point_clouds,
    )

    geometries = []

    for scanner_pose in poses:
        for camera in scanner_pose.cameras:
            camera_key = extract_camera_image_key(
                camera.image_file
            )

            if (
                segmentation_data is not None
                and only_segmented_images
                and camera_key not in segmentation_data.camera_keys
            ):
                continue

            geometry = create_camera_image_plane_geometry(
                camera=camera,
                image_dir=image_path,
                segmentation_data=segmentation_data,
                depth=depth,
                image_resolution=image_resolution,
                image_mode=image_mode,
                segmentation_opacity=segmentation_opacity,
            )

            geometries.append(
                geometry
            )

    return geometries


def create_camera_image_plane_geometry(
    camera: CameraPose,
    image_dir: str | Path,
    segmentation_data: CocoSegmentationData | None = None,
    depth: float = 1.0,
    image_resolution: int = 64,
    image_mode: Literal["color", "grayscale"] = "color",
    segmentation_opacity: float = 0.65,
) -> o3d.geometry.TriangleMesh:
    """Create one coloured 3D camera image plane.

    Args:
        camera: Camera pose and pinhole intrinsics.
        image_dir: Directory containing the camera image.
        segmentation_data: Optional filtered COCO segmentation data.
        depth: Distance of the image plane from the camera centre.
        image_resolution: Samples along the longest image dimension.
        image_mode: Display image in color or grayscale.
        segmentation_opacity: Opacity of segmentation overlays.

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
        camera=camera,
        image_path=image_path,
        pixels_uv=pixels_uv,
    )

    colors = _apply_image_mode(
        colors,
        image_mode=image_mode,
    )

    if segmentation_data is not None:
        colors = _apply_segmentation_overlays(
            colors=colors,
            camera=camera,
            pixels_uv=pixels_uv,
            sample_width=sample_width,
            sample_height=sample_height,
            segmentation_data=segmentation_data,
            opacity=segmentation_opacity,
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
    camera: CameraPose,
    image_path: Path,
    pixels_uv: np.ndarray,
) -> np.ndarray:
    """Sample RGB values from an oriented image at UV coordinates."""
    image = np.asarray(
        o3d.io.read_image(
            str(image_path)
        )
    )

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"Expected RGB image: {image_path}"
        )

    image = orient_camera_array(
        image,
        camera.image_file,
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


def _apply_image_mode(
    colors: np.ndarray,
    image_mode: str,
) -> np.ndarray:
    """Apply color or grayscale display mode."""
    if image_mode == "color":
        return colors

    if image_mode == "grayscale":
        luminance = (
            0.299 * colors[:, 0]
            + 0.587 * colors[:, 1]
            + 0.114 * colors[:, 2]
        )

        return np.column_stack(
            (
                luminance,
                luminance,
                luminance,
            )
        )

    raise ValueError(
        "image_mode must be 'color' or 'grayscale'."
    )


def _apply_segmentation_overlays(
    colors: np.ndarray,
    camera: CameraPose,
    pixels_uv: np.ndarray,
    sample_width: int,
    sample_height: int,
    segmentation_data: CocoSegmentationData,
    opacity: float,
) -> np.ndarray:
    """Overlay COCO segmentations on sampled image colours."""
    camera_key = extract_camera_image_key(
        camera.image_file
    )

    image_data = (
        segmentation_data.images_by_camera_key.get(
            camera_key
        )
    )

    if image_data is None:
        return colors

    image_id = image_data["id"]

    annotations = (
        segmentation_data.annotations_by_image_id.get(
            image_id,
            [],
        )
    )

    output_colors = colors.copy()

    dark_edge = np.array(
        [0.03, 0.03, 0.03],
        dtype=np.float64,
    )

    for annotation in annotations:
        mask = decode_coco_segmentation(
            annotation,
            image_height=image_data["height"],
            image_width=image_data["width"],
        )

        mask = orient_camera_array(
            mask,
            camera.image_file,
        )

        sampled_mask = sample_mask(
            mask,
            pixels_uv,
            sample_width,
            sample_height,
        )

        if not np.any(sampled_mask):
            continue

        boundary = find_mask_boundary(
            sampled_mask
        )

        mask_flat = sampled_mask.ravel()
        boundary_flat = boundary.ravel()

        segmentation_color = _get_segmentation_color(
            annotation["id"]
        )

        output_colors[mask_flat] = (
            (1.0 - opacity)
            * output_colors[mask_flat]
            + opacity
            * segmentation_color
        )

        output_colors[boundary_flat] = dark_edge

    return output_colors


def _get_segmentation_color(
    annotation_id: int,
) -> np.ndarray:
    """Generate a deterministic colour for one segmentation instance."""
    golden_ratio = 0.618033988749895

    hue = (
        annotation_id
        * golden_ratio
    ) % 1.0

    red, green, blue = colorsys.hsv_to_rgb(
        hue,
        0.75,
        1.0,
    )

    return np.array(
        [
            red,
            green,
            blue,
        ],
        dtype=np.float64,
    )


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
