"""Utilities for matching and orienting camera images and masks."""

import re

import numpy as np
from numpy.typing import NDArray


_CAMERA_IMAGE_PATTERN = re.compile(
    r"(pose(?P<pose>\d+)_image(?P<image>\d+))",
    re.IGNORECASE,
)


def extract_camera_image_key(
    file_name: str,
) -> str:
    """Extract the canonical camera-image key from a file name.

    Examples:
        ``pose4_image5.jpg`` becomes ``pose4_image5``.

        ``pose83_image5_jpg.rf.XAXlz7Dj22m5TpjbklEX.jpg``
        becomes ``pose83_image5``.

    Args:
        file_name: Image file name or path.

    Returns:
        Canonical camera-image key.

    Raises:
        ValueError: If no pose/image identifier can be found.
    """
    match = _CAMERA_IMAGE_PATTERN.search(
        file_name
    )

    if match is None:
        raise ValueError(
            "Could not find a camera-image identifier in "
            f"'{file_name}'."
        )

    return match.group(1).lower()


def extract_camera_image_number(
    file_name: str,
) -> int:
    """Extract the cube-map image number from a file name.

    Args:
        file_name: Image file name or path.

    Returns:
        Camera image number.

    Raises:
        ValueError: If no pose/image identifier can be found.
    """
    match = _CAMERA_IMAGE_PATTERN.search(
        file_name
    )

    if match is None:
        raise ValueError(
            "Could not find a camera-image identifier in "
            f"'{file_name}'."
        )

    return int(
        match.group("image")
    )


def orient_camera_array(
    array: NDArray,
    file_name: str,
) -> NDArray:
    """Orient an image or mask for camera-plane mapping.

    The same transformation is applied to RGB images and
    segmentation masks.

    Camera images 1 to 4 are rotated 180 degrees and then mirrored
    about the vertical image axis.

    Camera images 5 and 6 are flipped along the scanner red-axis
    direction established during camera-plane validation.

    Args:
        array: Image or mask array. This may be a 2D mask or a
            multi-channel image.
        file_name: Camera image file name.

    Returns:
        Oriented contiguous NumPy array.

    Raises:
        ValueError: If the image number is outside the expected
            cube-map range.
    """
    image_number = extract_camera_image_number(
        file_name
    )

    oriented = np.asarray(array)

    if 1 <= image_number <= 4:
        oriented = np.rot90(
            oriented,
            2,
            axes=(0, 1),
        )

        oriented = np.flip(
            oriented,
            axis=1,
        )

    elif image_number in (5, 6):
        oriented = np.flip(
            oriented,
            axis=0,
        )

    else:
        raise ValueError(
            "Expected camera image number between 1 and 6, "
            f"received {image_number}."
        )

    return np.ascontiguousarray(
        oriented
    )
