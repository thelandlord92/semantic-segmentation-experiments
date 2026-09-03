import json
from pathlib import Path

# Parent dir of the current file
parent_dir = Path(__file__).parent.parent.parent

# Data folder directory
data_dir = parent_dir / "data" / "sam_segmentations"

# Input files
beams_json_path = data_dir / "timber_beams_annotations.coco.json"
columns_json_path = data_dir / "timber_columns_annotations.coco.json"

# Output file
output_json_path = data_dir / "_annotations.coco.json"

# Load JSON files
with open(beams_json_path, "r", encoding="utf-8") as f:
    beams_data = json.load(f)

with open(columns_json_path, "r", encoding="utf-8") as f:
    columns_data = json.load(f)

# Build image lookup from the first file using file_name
beams_images_by_name = {
    image["file_name"]: image for image in beams_data["images"]
}

# Make sure all images in the second file exist in the first file
for image in columns_data["images"]:
    if image["file_name"] not in beams_images_by_name:
        raise ValueError(
            f"Image '{image['file_name']}' exists in columns JSON but not in beams JSON."
        )

# Build a mapping from old image_id in columns JSON to image_id in beams JSON
columns_image_id_to_beams_image_id = {}
for image in columns_data["images"]:
    file_name = image["file_name"]
    columns_image_id_to_beams_image_id[image["id"]] = beams_images_by_name[file_name]["id"]

# Standardize categories
merged_categories = [
    {"id": 1, "name": "timber_beams", "supercategory": "object"},
    {"id": 2, "name": "timber_columns", "supercategory": "object"},
]

# Force beams annotations to category_id = 1
for annotation in beams_data["annotations"]:
    annotation["category_id"] = 1

# Rebuild columns annotations with:
# - category_id = 2
# - remapped image_id
# - new unique annotation IDs
max_annotation_id = max(annotation["id"] for annotation in beams_data["annotations"]) \
    if beams_data["annotations"] else 0

new_columns_annotations = []
next_annotation_id = max_annotation_id + 1

for annotation in columns_data["annotations"]:
    new_annotation = annotation.copy()
    new_annotation["id"] = next_annotation_id
    new_annotation["category_id"] = 2
    new_annotation["image_id"] = columns_image_id_to_beams_image_id[annotation["image_id"]]
    new_columns_annotations.append(new_annotation)
    next_annotation_id += 1

# Build merged dataset
merged_data = {
    "images": beams_data["images"],
    "annotations": beams_data["annotations"] + new_columns_annotations,
    "categories": merged_categories,
}

# Keep optional COCO fields if present
for optional_key in ["info", "licenses"]:
    if optional_key in beams_data:
        merged_data[optional_key] = beams_data[optional_key]

# Save merged JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2)

print(f"Merged COCO JSON saved to: {output_json_path}")