"""Geometry utilities for COCO segmentation masks."""

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as mask_utils


def decode_coco_segmentation(
    annotation: dict,
    image_height: int,
    image_width: int,
) -> NDArray[np.bool_]:
    """Decode a COCO segmentation into a binary mask.

    Supports polygon, uncompressed RLE, and compressed RLE
    segmentations.

    Args:
        annotation: COCO annotation dictionary.
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Binary segmentation mask.
    """
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
