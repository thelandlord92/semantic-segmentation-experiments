import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pathlib import Path

# -----------------------------
# DATASET DIRECTORY
# -----------------------------
parent_dir = Path(__file__).parent.parent.parent
dataset_dir = parent_dir / "data" / "sam_segmentations" / "hxe"

# -----------------------------
# CONFIG
# -----------------------------
coco_json_path = dataset_dir / "roboflow_dataset" / "_annotations.coco.json"
images_dir = dataset_dir / "roboflow_dataset"
output_dir = dataset_dir / "mask_visuals"

# -----------------------------
# VISUALISATION CONFIG
# -----------------------------

# Mask transparency
# 0.0 = completely transparent
# 1.0 = fully opaque
mask_alpha = 0.45

# Toggle bounding boxes and labels
show_box_and_label = False

# Toggle mask contours
show_contours = True

# Convert background image to grayscale
# Masks, contours, boxes and labels remain coloured
grayscale = True

# Image contrast
# 1.0 = original
# >1.0 = more contrast
# <1.0 = less contrast
contrast = 1.0

# Image brightness
# 0 = original
# >0 = brighter
# <0 = darker
brightness = 0

# -----------------------------
# CREATE OUTPUT DIRECTORY
# -----------------------------
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD COCO JSON
# -----------------------------
with open(coco_json_path, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

categories = {
    cat["id"]: cat["name"]
    for cat in coco["categories"]
}

# -----------------------------
# GROUP ANNOTATIONS BY IMAGE ID
# -----------------------------
anns_by_image = {}

for ann in annotations:
    anns_by_image.setdefault(
        ann["image_id"],
        []
    ).append(ann)


# -----------------------------
# FUNCTIONS
# -----------------------------

def get_color(idx):
    """
    Generate a deterministic colour for an annotation.

    The same index will always produce the same colour.
    """

    np.random.seed(idx + 123)

    color = np.random.randint(
        0,
        255,
        size=3
    ).tolist()

    return tuple(int(c) for c in color)


def adjust_brightness_contrast(
    image,
    contrast=1.0,
    brightness=0
):
    """
    Adjust image brightness and contrast.

    contrast:
        1.0 = unchanged
        >1.0 = increase contrast
        <1.0 = decrease contrast

    brightness:
        0 = unchanged
        positive = brighter
        negative = darker
    """

    adjusted = cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=brightness
    )

    return adjusted


def convert_to_grayscale(image):
    """
    Convert the image to grayscale while retaining
    three colour channels.

    Keeping three channels allows coloured masks,
    contours, bounding boxes and labels to be drawn
    on top of the grayscale image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray_bgr = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    return gray_bgr


def decode_segmentation(
    segmentation,
    height,
    width
):
    """
    Decode COCO polygon or RLE segmentation
    into a binary mask.
    """

    if isinstance(segmentation, list):

        # -----------------------------
        # POLYGON FORMAT
        # -----------------------------
        rles = maskUtils.frPyObjects(
            segmentation,
            height,
            width
        )

        rle = maskUtils.merge(rles)

        mask = maskUtils.decode(rle)

    elif isinstance(segmentation, dict):

        # -----------------------------
        # RLE FORMAT
        # -----------------------------

        if isinstance(
            segmentation["counts"],
            list
        ):

            # Uncompressed RLE
            rle = maskUtils.frPyObjects(
                segmentation,
                height,
                width
            )

            mask = maskUtils.decode(rle)

        else:

            # Compressed RLE
            mask = maskUtils.decode(
                segmentation
            )

    else:

        raise ValueError(
            "Unknown segmentation format"
        )

    # Some polygon segmentations may return
    # multiple mask channels
    if mask.ndim == 3:

        mask = np.any(
            mask,
            axis=2
        ).astype(np.uint8)

    return mask.astype(np.uint8)


# -----------------------------
# LOOP THROUGH IMAGES
# -----------------------------
for img_info in images:

    file_name = img_info["file_name"]
    image_id = img_info["id"]
    height = img_info["height"]
    width = img_info["width"]

    # -----------------------------
    # IMAGE PATH
    # -----------------------------
    img_path = os.path.join(
        images_dir,
        file_name
    )

    if not os.path.exists(img_path):

        print(
            f"Image not found: {img_path}"
        )

        continue

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------
    image = cv2.imread(img_path)

    if image is None:

        print(
            f"Could not read image: {img_path}"
        )

        continue

    # -----------------------------
    # CONVERT TO GRAYSCALE
    # -----------------------------
    if grayscale:

        image = convert_to_grayscale(
            image
        )

    # -----------------------------
    # BRIGHTNESS AND CONTRAST
    # -----------------------------
    image = adjust_brightness_contrast(
        image,
        contrast=contrast,
        brightness=brightness
    )

    # Image onto which masks will be drawn
    overlay = image.copy()

    # Get all annotations belonging
    # to this image
    anns = anns_by_image.get(
        image_id,
        []
    )

    # -----------------------------
    # PROCESS ANNOTATIONS
    # -----------------------------
    for i, ann in enumerate(anns):

        category_id = ann[
            "category_id"
        ]

        label = categories.get(
            category_id,
            str(category_id)
        )

        # Generate colour for instance
        color = get_color(
            i + category_id * 100
        )

        # -----------------------------
        # DECODE MASK
        # -----------------------------
        mask = decode_segmentation(
            ann["segmentation"],
            height,
            width
        )

        # -----------------------------
        # CREATE COLOURED MASK
        # -----------------------------
        colored_mask = np.zeros_like(
            image,
            dtype=np.uint8
        )

        colored_mask[
            mask == 1
        ] = color

        # -----------------------------
        # OVERLAY MASK
        # -----------------------------
        overlay = cv2.addWeighted(
            overlay,
            1.0,
            colored_mask,
            mask_alpha,
            0
        )

        # -----------------------------
        # DRAW MASK CONTOUR
        # -----------------------------
        if show_contours:

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                color,
                2
            )

        # -----------------------------
        # DRAW BOX AND LABEL
        # -----------------------------
        if (
            show_box_and_label
            and "bbox" in ann
        ):

            x, y, w, h = map(
                int,
                ann["bbox"]
            )

            # Bounding box
            cv2.rectangle(
                overlay,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Label
            cv2.putText(
                overlay,
                label,
                (
                    x,
                    max(20, y - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

    # -----------------------------
    # SAVE OUTPUT IMAGE
    # -----------------------------
    out_path = os.path.join(
        output_dir,
        file_name
    )

    cv2.imwrite(
        out_path,
        overlay
    )

    print(
        f"Saved: {out_path} | "
        f"Annotations: {len(anns)}"
    )

print("Done.")