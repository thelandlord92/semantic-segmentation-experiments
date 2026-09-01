"""
Merge two COCO JSON files, optionally simplify polygon segmentations,
and optionally plot the simplified polygons onto matching source images.
"""

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from simplify_polyline import simplify, vw_simplify


# =============================================================================
# USER INPUTS
# =============================================================================

# Parent directory of the current file.
PARENT_DIR = Path(__file__).parent.parent.parent

# Input COCO JSON files.
BEAMS_JSON_PATH = PARENT_DIR / "results/beams_annotations.coco.json"
COLUMNS_JSON_PATH = PARENT_DIR / "results/columns_annotations.coco.json"

# Output merged COCO JSON file.
OUTPUT_JSON_PATH = PARENT_DIR / "results/_annotations.coco.json"

# Class names in the merged COCO JSON.
BEAM_CLASS_NAME = "timber beams"
COLUMN_CLASS_NAME = "timber columns"

# Simplification settings.
SIMPLIFY_POLYGONS = True

# Options:
# "dp" = Douglas-Peucker style simplification
# "vw" = Visvalingam-Whyatt style simplification
SIMPLIFY_METHOD = "dp"

# Used when SIMPLIFY_METHOD = "dp".
# Larger values remove more vertices.
MIN_DIST = 4.0

# Used when SIMPLIFY_METHOD = "vw".
# Larger values remove more vertices.
MIN_AREA = 10.0

# Plot settings.
PLOT_SIMPLIFIED_POLYGONS = True

# Folder containing the original source images.
SOURCE_IMAGE_DIR = PARENT_DIR / "data/images/hxe_poses"

# Folder where plotted polygon check images will be saved.
PLOT_DIR = PARENT_DIR / "results/coco_polygon_checks_4.0_rev1"


# =============================================================================
# JSON HELPERS
# =============================================================================

def load_json(json_path):
    """Load a JSON file."""
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, json_path):
    """Save a JSON file."""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# =============================================================================
# POLYGON HELPERS
# =============================================================================

def polygon_to_points(polygon):
    """Convert a flat COCO polygon list to an Nx2 NumPy array."""
    return np.array(polygon, dtype=np.float64).reshape(-1, 2)


def points_to_polygon(points):
    """Convert an Nx2 array back to a flat COCO polygon list."""
    return np.asarray(points, dtype=np.float64).reshape(-1).tolist()


def polygon_area(polygon):
    """Calculate polygon area using the shoelace formula."""
    points = polygon_to_points(polygon)

    if len(points) < 3:
        return 0.0

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    area = 0.5 * abs(
        np.dot(x_coords, np.roll(y_coords, -1))
        - np.dot(y_coords, np.roll(x_coords, -1))
    )

    return float(area)


def bbox_from_polygons(polygons):
    """Calculate COCO bbox [x, y, width, height] from polygon lists."""
    valid_points = []

    for polygon in polygons:
        points = polygon_to_points(polygon)

        if len(points) >= 3:
            valid_points.append(points)

    if not valid_points:
        return None

    points = np.vstack(valid_points)

    x_min = float(points[:, 0].min())
    y_min = float(points[:, 1].min())
    x_max = float(points[:, 0].max())
    y_max = float(points[:, 1].max())

    return [
        x_min,
        y_min,
        x_max - x_min,
        y_max - y_min,
    ]


def simplify_polygon(polygon):
    """
    Simplify one COCO polygon.

    The polygon is expected in COCO format:
    [x1, y1, x2, y2, ...]
    """
    if len(polygon) < 6:
        return polygon

    points = polygon_to_points(polygon)

    if len(points) < 3:
        return polygon

    if SIMPLIFY_METHOD == "dp":
        simplified = simplify_with_fallback(points)
    elif SIMPLIFY_METHOD == "vw":
        simplified = vw_simplify_with_fallback(points)
    else:
        raise ValueError(f"Unsupported simplification method: {SIMPLIFY_METHOD}")

    simplified = np.asarray(simplified, dtype=np.float64)

    if simplified.ndim != 2 or simplified.shape[0] < 3:
        return polygon

    return points_to_polygon(simplified)


def simplify_with_fallback(points):
    """Run simplify-polyline's simplify function with a robust fallback."""
    try:
        return simplify(points, min_dist=MIN_DIST, is_closed=True)
    except TypeError:
        return simplify(points, MIN_DIST, True)


def vw_simplify_with_fallback(points):
    """Run simplify-polyline's Visvalingam-Whyatt simplification."""
    try:
        return vw_simplify(points, min_area=MIN_AREA, is_closed=True)
    except TypeError:
        return vw_simplify(points, MIN_AREA, True)


def simplify_annotation(annotation):
    """
    Simplify all polygon segmentations in one COCO annotation.

    RLE segmentations are left unchanged.
    """
    new_annotation = copy.deepcopy(annotation)
    segmentation = new_annotation.get("segmentation")

    if not isinstance(segmentation, list):
        return new_annotation

    simplified_polygons = []

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue

        simplified_polygon = simplify_polygon(polygon)

        if len(simplified_polygon) >= 6:
            simplified_polygons.append(simplified_polygon)

    if not simplified_polygons:
        return None

    new_annotation["segmentation"] = simplified_polygons
    new_annotation["area"] = sum(
        polygon_area(polygon) for polygon in simplified_polygons
    )

    bbox = bbox_from_polygons(simplified_polygons)
    if bbox is None:
        return None

    new_annotation["bbox"] = bbox

    return new_annotation


# =============================================================================
# MERGE HELPERS
# =============================================================================

def merge_coco_files(beams_data, columns_data):
    """Merge beam and column COCO JSON files into one COCO dataset."""
    beams_images_by_name = {
        image["file_name"]: image for image in beams_data["images"]
    }

    for image in columns_data["images"]:
        file_name = image["file_name"]

        if file_name not in beams_images_by_name:
            raise ValueError(
                f"Image '{file_name}' exists in columns JSON, "
                "but not in beams JSON."
            )

    columns_image_id_to_beams_image_id = {}

    for image in columns_data["images"]:
        file_name = image["file_name"]
        columns_image_id_to_beams_image_id[image["id"]] = (
            beams_images_by_name[file_name]["id"]
        )

    merged_categories = [
        {
            "id": 1,
            "name": BEAM_CLASS_NAME,
            "supercategory": "object",
        },
        {
            "id": 2,
            "name": COLUMN_CLASS_NAME,
            "supercategory": "object",
        },
    ]

    merged_annotations = []
    next_annotation_id = 1

    for annotation in beams_data["annotations"]:
        new_annotation = copy.deepcopy(annotation)
        new_annotation["id"] = next_annotation_id
        new_annotation["category_id"] = 1

        if SIMPLIFY_POLYGONS:
            new_annotation = simplify_annotation(new_annotation)

        if new_annotation is not None:
            merged_annotations.append(new_annotation)
            next_annotation_id += 1

    for annotation in columns_data["annotations"]:
        new_annotation = copy.deepcopy(annotation)
        new_annotation["id"] = next_annotation_id
        new_annotation["category_id"] = 2
        new_annotation["image_id"] = columns_image_id_to_beams_image_id[
            annotation["image_id"]
        ]

        if SIMPLIFY_POLYGONS:
            new_annotation = simplify_annotation(new_annotation)

        if new_annotation is not None:
            merged_annotations.append(new_annotation)
            next_annotation_id += 1

    merged_data = {
        "images": beams_data["images"],
        "annotations": merged_annotations,
        "categories": merged_categories,
    }

    for optional_key in ["info", "licenses"]:
        if optional_key in beams_data:
            merged_data[optional_key] = beams_data[optional_key]

    return merged_data


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def check_source_images(coco_data):
    """Check that all image names in the COCO file exist in the source folder."""
    missing_images = []

    for image_record in coco_data["images"]:
        image_path = SOURCE_IMAGE_DIR / image_record["file_name"]

        if not image_path.exists():
            missing_images.append(image_record["file_name"])

    if missing_images:
        missing_text = "\n".join(missing_images)
        raise FileNotFoundError(
            "Some images in the COCO JSON were not found in the "
            f"source folder:\n{missing_text}"
        )


def color_from_category_id(category_id):
    """Generate a deterministic RGB color from a category ID."""
    rng = np.random.default_rng(category_id)
    return tuple(rng.integers(0, 256, size=3).tolist())


def draw_polygons_on_images(coco_data):
    """Draw COCO polygons onto matching source images."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    check_source_images(coco_data)

    annotations_by_image_id = defaultdict(list)

    for annotation in coco_data["annotations"]:
        annotations_by_image_id[annotation["image_id"]].append(annotation)

    categories_by_id = {
        category["id"]: category["name"]
        for category in coco_data["categories"]
    }

    for image_record in coco_data["images"]:
        image_path = SOURCE_IMAGE_DIR / image_record["file_name"]

        image = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        annotations = annotations_by_image_id.get(image_record["id"], [])

        for annotation in annotations:
            category_id = annotation["category_id"]
            category_name = categories_by_id.get(category_id, "unknown")
            color = color_from_category_id(category_id)

            fill = (*color, 70)
            outline = (*color, 255)

            segmentation = annotation.get("segmentation", [])

            if not isinstance(segmentation, list):
                continue

            for polygon in segmentation:
                if not isinstance(polygon, list) or len(polygon) < 6:
                    continue

                points = polygon_to_points(polygon)
                point_tuples = [tuple(point) for point in points]

                draw.polygon(point_tuples, fill=fill)
                draw.line(
                    point_tuples + [point_tuples[0]],
                    fill=outline,
                    width=3,
                )

                draw.text(
                    point_tuples[0],
                    category_name,
                    fill=outline,
                )

        plotted = Image.alpha_composite(image, overlay).convert("RGB")
        output_path = PLOT_DIR / f"{Path(image_record['file_name']).stem}_plot.png"
        plotted.save(output_path)

        print(f"Saved plot: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the COCO merge, optional simplification, and optional plotting."""
    beams_data = load_json(BEAMS_JSON_PATH)
    columns_data = load_json(COLUMNS_JSON_PATH)

    merged_data = merge_coco_files(
        beams_data=beams_data,
        columns_data=columns_data,
    )

    save_json(merged_data, OUTPUT_JSON_PATH)

    print(f"Saved merged COCO JSON: {OUTPUT_JSON_PATH}")
    print(f"Images: {len(merged_data['images'])}")
    print(f"Annotations: {len(merged_data['annotations'])}")
    print(f"Categories: {len(merged_data['categories'])}")

    if PLOT_SIMPLIFIED_POLYGONS:
        draw_polygons_on_images(merged_data)


if __name__ == "__main__":
    main()