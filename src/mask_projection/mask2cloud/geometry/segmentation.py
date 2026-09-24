"""Geometry utilities for COCO segmentation masks."""

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as mask_utils

from ..geometry.image_mapping import (
    extract_camera_image_key,
    orient_camera_array,
)
from ..models import CameraPose, CocoSegmentationData


def decode_coco_segmentation(
    annotation: dict,
    image_height: int,
    image_width: int,
) -> NDArray[np.bool_]:
    """Decode a COCO segmentation into a binary mask."""
    segmentation = annotation["segmentation"]

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(
            segmentation,
            image_height,
            image_width,
        )

        rle = mask_utils.merge(
            rles
        )

    elif isinstance(segmentation, dict):
        if isinstance(
            segmentation.get("counts"),
            list,
        ):
            rle = mask_utils.frPyObjects(
                segmentation,
                image_height,
                image_width,
            )
        else:
            rle = segmentation

    else:
        raise TypeError(
            "Unsupported COCO segmentation format."
        )

    mask = mask_utils.decode(
        rle
    )

    if mask.ndim == 3:
        mask = np.any(
            mask,
            axis=2,
        )

    return mask.astype(
        bool
    )


def build_camera_segmentation_masks(
    camera: CameraPose,
    segmentation_data: CocoSegmentationData,
    segmentation_mode: str,
) -> list[dict[str, Any]]:
    """Build semantic or instance masks for one camera.

    Returns a list of dictionaries. Each dictionary contains:
    - mask: oriented binary mask
    - category_id: COCO category ID
    - category_name: COCO category name
    - annotation_id: original annotation ID for instance mode,
      or None for semantic mode
    """
    camera_key = extract_camera_image_key(
        camera.image_file
    )

    image_data = (
        segmentation_data.images_by_camera_key.get(
            camera_key
        )
    )

    if image_data is None:
        return []

    image_id = image_data["id"]

    annotations = (
        segmentation_data.annotations_by_image_id.get(
            image_id,
            [],
        )
    )

    if segmentation_mode == "instance":
        return _build_instance_masks(
            camera=camera,
            image_data=image_data,
            annotations=annotations,
            segmentation_data=segmentation_data,
        )

    if segmentation_mode == "semantic":
        return _build_semantic_masks(
            camera=camera,
            image_data=image_data,
            annotations=annotations,
            segmentation_data=segmentation_data,
        )

    raise ValueError(
        "segmentation_mode must be 'semantic' or 'instance'."
    )


def sample_mask(
    mask: NDArray[np.bool_],
    pixels_uv: NDArray[np.float64],
    sample_width: int,
    sample_height: int,
) -> NDArray[np.bool_]:
    """Sample a mask using image-plane UV coordinates."""
    u_indices = np.rint(
        pixels_uv[:, 0]
    ).astype(np.int64)

    v_indices = np.rint(
        pixels_uv[:, 1]
    ).astype(np.int64)

    u_indices = np.clip(
        u_indices,
        0,
        mask.shape[1] - 1,
    )

    v_indices = np.clip(
        v_indices,
        0,
        mask.shape[0] - 1,
    )

    sampled = mask[
        v_indices,
        u_indices,
    ]

    return sampled.reshape(
        sample_height,
        sample_width,
    )


def find_mask_boundary(
    sampled_mask: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Find a one-sample-wide internal mask boundary."""
    padded = np.pad(
        sampled_mask,
        1,
        mode="constant",
        constant_values=False,
    )

    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )

    return (
        sampled_mask
        & ~eroded
    )


def _build_instance_masks(
    camera: CameraPose,
    image_data: dict[str, Any],
    annotations: list[dict[str, Any]],
    segmentation_data: CocoSegmentationData,
) -> list[dict[str, Any]]:
    """Build one mask per annotation instance."""
    mask_records = []

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

        category_id = annotation["category_id"]

        mask_records.append(
            {
                "mask": mask,
                "category_id": category_id,
                "category_name": segmentation_data.categories_by_id[
                    category_id
                ],
                "annotation_id": annotation["id"],
            }
        )

    return mask_records


def _build_semantic_masks(
    camera: CameraPose,
    image_data: dict[str, Any],
    annotations: list[dict[str, Any]],
    segmentation_data: CocoSegmentationData,
) -> list[dict[str, Any]]:
    """Build one merged mask per category."""
    masks_by_category: dict[int, NDArray[np.bool_]] = {}
    names_by_category: dict[int, str] = {}

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

        category_id = annotation["category_id"]

        if category_id not in masks_by_category:
            masks_by_category[category_id] = mask.copy()
        else:
            masks_by_category[category_id] |= mask

        names_by_category[category_id] = (
            segmentation_data.categories_by_id[
                category_id
            ]
        )

    return [
        {
            "mask": mask,
            "category_id": category_id,
            "category_name": names_by_category[category_id],
            "annotation_id": None,
        }
        for category_id, mask
        in masks_by_category.items()
    ]