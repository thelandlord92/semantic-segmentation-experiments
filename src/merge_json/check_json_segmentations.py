"""
Check one COCO JSON file for annotations with multi-polygon segmentations.

A multi-polygon annotation is an annotation where:
annotation["segmentation"] is a list, and len(annotation["segmentation"]) > 1
"""

import json
from pathlib import Path


# =============================================================================
# USER INPUT
# =============================================================================

COCO_JSON_PATH = Path("results/_annotations.coco.json")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Check one COCO JSON file for multi-polygon annotations."""
    if not COCO_JSON_PATH.exists():
        print(f"File not found: {COCO_JSON_PATH}")
        return

    with open(COCO_JSON_PATH, "r", encoding="utf-8") as file:
        coco_data = json.load(file)

    annotations = coco_data.get("annotations", [])

    total_annotations = len(annotations)
    polygon_annotations = 0
    multi_polygon_annotations = 0
    non_polygon_annotations = 0

    for annotation in annotations:
        segmentation = annotation.get("segmentation")

        if isinstance(segmentation, list):
            polygon_annotations += 1

            if len(segmentation) > 1:
                multi_polygon_annotations += 1
        else:
            non_polygon_annotations += 1

    if total_annotations > 0:
        percentage = (multi_polygon_annotations / total_annotations) * 100
    else:
        percentage = 0.0

    print("=" * 70)
    print("COCO SEGMENTATION CHECK")
    print("=" * 70)
    print(f"File:                                {COCO_JSON_PATH}")
    print("-" * 70)
    print(f"Total annotations:                   {total_annotations}")
    print(f"Polygon annotations:                 {polygon_annotations}")
    print(f"Non-polygon annotations:             {non_polygon_annotations}")
    print(f"Annotations with multiple polygons:  {multi_polygon_annotations}")
    print(f"Percentage multi-polygon:            {percentage:.2f}%")
    print("=" * 70)

    if multi_polygon_annotations == 0:
        print("Result: No multi-polygon annotations found.")
        print("Likely structure: one polygon = one annotation.")
    else:
        print("Result: Multi-polygon annotations found.")
        print("Likely structure: some annotations contain multiple polygons.")


if __name__ == "__main__":
    main()