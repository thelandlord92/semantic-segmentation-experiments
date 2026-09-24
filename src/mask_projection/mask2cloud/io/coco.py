"""Functions for loading COCO segmentation data."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..geometry.image_mapping import (
    extract_camera_image_key,
)
from ..models import CocoSegmentationData


def load_coco_segmentations(
    file_path: str | Path,
    segmentation_categories: str | Sequence[str] | None = None,
) -> CocoSegmentationData:
    """Load and filter COCO segmentation data.

    Images without segmentations in the selected categories are
    excluded.

    Args:
        file_path: Path to the COCO JSON file.
        segmentation_categories: Category names to retain. If None,
            annotations from all categories are retained.

    Returns:
        Filtered COCO segmentation data.

    Raises:
        FileNotFoundError: If the COCO file does not exist.
        ValueError: If requested categories are unavailable.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"COCO file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco_data = json.load(file)

    categories_by_id = {
        category["id"]: category["name"]
        for category in coco_data["categories"]
    }

    selected_category_ids = _select_category_ids(
        categories_by_id,
        segmentation_categories,
    )

    annotations_by_image_id = (
        _group_selected_annotations(
            coco_data["annotations"],
            selected_category_ids,
        )
    )

    annotated_image_ids = set(
        annotations_by_image_id
    )

    images_by_id: dict[int, dict[str, Any]] = {}
    images_by_camera_key: dict[str, dict[str, Any]] = {}

    for image_data in coco_data["images"]:
        image_id = image_data["id"]

        if image_id not in annotated_image_ids:
            continue

        camera_key = extract_camera_image_key(
            image_data["file_name"]
        )

        if camera_key in images_by_camera_key:
            previous_file = images_by_camera_key[
                camera_key
            ]["file_name"]

            raise ValueError(
                "Multiple COCO images resolve to "
                f"'{camera_key}': '{previous_file}' and "
                f"'{image_data['file_name']}'."
            )

        images_by_id[image_id] = image_data
        images_by_camera_key[camera_key] = image_data

    return CocoSegmentationData(
        categories_by_id=categories_by_id,
        images_by_id=images_by_id,
        images_by_camera_key=images_by_camera_key,
        annotations_by_image_id=annotations_by_image_id,
        selected_category_ids=selected_category_ids,
    )


def _select_category_ids(
    categories_by_id: dict[int, str],
    segmentation_categories: str | Sequence[str] | None,
) -> set[int]:
    """Resolve requested category names to COCO category IDs."""
    if segmentation_categories is None:
        return set(
            categories_by_id
        )

    if isinstance(
        segmentation_categories,
        str,
    ):
        requested_categories = [
            segmentation_categories
        ]
    else:
        requested_categories = list(
            segmentation_categories
        )

    if not requested_categories:
        raise ValueError(
            "segmentation_categories cannot be empty."
        )

    categories_by_name = {
        name.casefold(): category_id
        for category_id, name
        in categories_by_id.items()
    }

    selected_ids = set()

    for category_name in requested_categories:
        normalized_name = (
            category_name
            .strip()
            .casefold()
        )

        if normalized_name not in categories_by_name:
            available = ", ".join(
                sorted(categories_by_name)
            )

            raise ValueError(
                f"Unknown segmentation category "
                f"'{category_name}'. Available categories: "
                f"{available}."
            )

        selected_ids.add(
            categories_by_name[
                normalized_name
            ]
        )

    return selected_ids


def _group_selected_annotations(
    annotations: list[dict[str, Any]],
    selected_category_ids: set[int],
) -> dict[int, list[dict[str, Any]]]:
    """Group valid selected annotations by COCO image ID."""
    annotations_by_image_id = {}

    for annotation in annotations:
        if (
            annotation["category_id"]
            not in selected_category_ids
        ):
            continue

        segmentation = annotation.get(
            "segmentation"
        )

        if segmentation in (
            None,
            [],
            {},
        ):
            continue

        image_id = annotation["image_id"]

        annotations_by_image_id.setdefault(
            image_id,
            [],
        ).append(
            annotation
        )

    return annotations_by_image_id
