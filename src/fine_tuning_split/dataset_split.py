import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

# Parent directory of current script
parent_dir = Path(__file__).parent.parent.parent

# Dataset directory
dataset_dir = parent_dir / "data" / "ground_truth" / "hxe"

# Input files
images_dir = dataset_dir / "images"
coco_json_path = dataset_dir / "_annotations.coco.json"

# Output directory
output_dir = dataset_dir / "split_dataset"

# ------------------------------------------------------------
# DATA SPLIT
# Order:
# [training %, validation %, testing %]
# ------------------------------------------------------------

split = [70, 15, 15]

# Random seed makes the split reproducible.
# Using the same seed produces the same split.
random_seed = 42

# Remove an existing split_dataset folder before creating
# a new split.
overwrite = True


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

split_names = ["train", "valid", "test"]

if len(split) != 3:
    raise ValueError(
        "split must contain exactly three values: "
        "[train, validation, test]"
    )

if any(value < 0 for value in split):
    raise ValueError(
        "Split percentages cannot be negative."
    )

if abs(sum(split) - 100) > 1e-9:
    raise ValueError(
        f"Split percentages must add up to 100. "
        f"Current total: {sum(split)}"
    )

if not images_dir.exists():
    raise FileNotFoundError(
        f"Images directory not found: {images_dir}"
    )

if not coco_json_path.exists():
    raise FileNotFoundError(
        f"COCO JSON not found: {coco_json_path}"
    )


# ============================================================
# LOAD COCO JSON
# ============================================================

with open(coco_json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])
categories = coco.get("categories", [])

if not images:
    raise ValueError(
        "No images were found in the COCO JSON."
    )

print(f"Total images: {len(images)}")
print(f"Total annotations: {len(annotations)}")


# ============================================================
# CHECK THAT ALL REFERENCED IMAGES EXIST
# ============================================================

missing_images = []

for image in images:
    image_path = images_dir / image["file_name"]

    if not image_path.exists():
        missing_images.append(image["file_name"])

if missing_images:
    print("\nMissing image files:")

    for file_name in missing_images[:20]:
        print(f"  {file_name}")

    raise FileNotFoundError(
        f"{len(missing_images)} image(s) referenced by the "
        f"COCO JSON were not found."
    )


# ============================================================
# CALCULATE SPLIT SIZES
# ============================================================

def calculate_split_counts(total_images, percentages):
    """
    Convert percentages into integer image counts while
    ensuring that:

    1. All images are assigned.
    2. Zero-percent splits remain zero.
    3. Rounding errors are handled consistently.
    """

    raw_counts = [
        total_images * percentage / 100
        for percentage in percentages
    ]

    counts = [
        int(value)
        for value in raw_counts
    ]

    remaining = total_images - sum(counts)

    # Splits eligible to receive remaining images.
    eligible_indices = [
        i
        for i, percentage in enumerate(percentages)
        if percentage > 0
    ]

    # Sort according to largest fractional remainder.
    eligible_indices.sort(
        key=lambda i: raw_counts[i] - counts[i],
        reverse=True
    )

    for i in range(remaining):
        target_index = eligible_indices[
            i % len(eligible_indices)
        ]

        counts[target_index] += 1

    return counts


split_counts = calculate_split_counts(
    len(images),
    split
)

train_count, valid_count, test_count = split_counts

print("\nRequested split:")
print(f"  Train: {split[0]}%")
print(f"  Valid: {split[1]}%")
print(f"  Test:  {split[2]}%")

print("\nActual image counts:")
print(f"  Train: {train_count}")
print(f"  Valid: {valid_count}")
print(f"  Test:  {test_count}")


# ============================================================
# RANDOMISE IMAGES
# ============================================================

rng = random.Random(random_seed)

shuffled_images = images.copy()
rng.shuffle(shuffled_images)


# ============================================================
# CREATE IMAGE SPLITS
# ============================================================

train_end = train_count
valid_end = train_end + valid_count

split_images = {
    "train": shuffled_images[:train_end],
    "valid": shuffled_images[train_end:valid_end],
    "test": shuffled_images[valid_end:]
}


# ============================================================
# GROUP ANNOTATIONS BY IMAGE ID
# ============================================================

annotations_by_image = defaultdict(list)

for annotation in annotations:
    annotations_by_image[
        annotation["image_id"]
    ].append(annotation)


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

if output_dir.exists():

    if overwrite:
        print(
            f"\nRemoving existing output directory: "
            f"{output_dir}"
        )

        shutil.rmtree(output_dir)

    else:
        raise FileExistsError(
            f"Output directory already exists: "
            f"{output_dir}"
        )

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CATEGORY LOOKUP FOR REPORTING
# ============================================================

category_names = {
    category["id"]: category["name"]
    for category in categories
}


# ============================================================
# CREATE EACH SPLIT
# ============================================================

for split_name in split_names:

    selected_images = split_images[split_name]

    split_dir = output_dir / split_name

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # REINDEX IMAGE IDs
    # --------------------------------------------------------

    image_id_mapping = {}

    new_images = []

    for new_image_id, image in enumerate(
        selected_images,
        start=1
    ):

        old_image_id = image["id"]

        image_id_mapping[
            old_image_id
        ] = new_image_id

        new_image = image.copy()
        new_image["id"] = new_image_id

        new_images.append(
            new_image
        )

        # Copy image into split directory.
        source_path = (
            images_dir
            / image["file_name"]
        )

        destination_path = (
            split_dir
            / image["file_name"]
        )

        shutil.copy2(
            source_path,
            destination_path
        )

    # --------------------------------------------------------
    # CREATE SPLIT ANNOTATIONS
    # --------------------------------------------------------

    new_annotations = []

    new_annotation_id = 1

    for image in selected_images:

        old_image_id = image["id"]

        image_annotations = (
            annotations_by_image.get(
                old_image_id,
                []
            )
        )

        for annotation in image_annotations:

            new_annotation = annotation.copy()

            # Assign new annotation ID.
            new_annotation["id"] = (
                new_annotation_id
            )

            # Map old image ID to the new image ID.
            new_annotation["image_id"] = (
                image_id_mapping[
                    old_image_id
                ]
            )

            new_annotations.append(
                new_annotation
            )

            new_annotation_id += 1

    # --------------------------------------------------------
    # CREATE NEW COCO JSON
    # --------------------------------------------------------

    # Preserve all top-level metadata from the original
    # JSON except images and annotations.
    split_coco = {
        key: value
        for key, value in coco.items()
        if key not in {
            "images",
            "annotations"
        }
    }

    split_coco["images"] = new_images
    split_coco["annotations"] = (
        new_annotations
    )

    # --------------------------------------------------------
    # SAVE COCO JSON
    # --------------------------------------------------------

    split_json_path = (
        split_dir
        / "_annotations.coco.json"
    )

    with open(
        split_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_coco,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # COUNT ANNOTATIONS BY CLASS
    # --------------------------------------------------------

    class_counts = defaultdict(int)

    for annotation in new_annotations:

        category_id = annotation[
            "category_id"
        ]

        class_counts[
            category_id
        ] += 1

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        f"\n{split_name.upper()}"
    )

    print(
        f"  Images: "
        f"{len(new_images)}"
    )

    print(
        f"  Annotations: "
        f"{len(new_annotations)}"
    )

    for category_id, category_name in (
        category_names.items()
    ):

        print(
            f"  {category_name}: "
            f"{class_counts[category_id]}"
        )

    print(
        f"  JSON: "
        f"{split_json_path}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

total_split_images = sum(
    len(value)
    for value in split_images.values()
)

assert total_split_images == len(images)

print("\n--------------------------------")
print("Dataset split completed.")
print("--------------------------------")
print(
    f"Random seed: {random_seed}"
)
print(
    f"Output: {output_dir}"
)