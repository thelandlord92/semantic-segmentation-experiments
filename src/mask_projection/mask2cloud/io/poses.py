"""Functions for reading scanner and camera pose data."""

import json
from pathlib import Path

import numpy as np

from ..models.pose import CameraPose, ScannerPose


def load_pose_data(
    file_path: str | Path,
) -> list[ScannerPose]:
    """Load scanner and camera poses from a pose-data JSON file.

    Args:
        file_path: Path to the pose-data JSON file.

    Returns:
        Scanner poses and their associated pinhole camera poses.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the expected pose structure is unavailable.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Pose-data file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        json_text = file.read()

    json_text = json_text.replace(
        '"box_size_m": [inf, inf, inf]',
        '"box_size_m": [null, null, null]',
    )

    data = json.loads(json_text)

    if "poses" not in data:
        raise ValueError(
            "Pose-data JSON does not contain a 'poses' field."
        )

    scanner_poses = []

    for pose_data in data["poses"]:
        scanner_poses.append(
            _parse_scanner_pose(pose_data)
        )

    return scanner_poses


def _parse_scanner_pose(
    pose_data: dict,
) -> ScannerPose:
    """Convert one JSON scanner pose to ScannerPose."""
    pose_id = pose_data["pose_id"]
    source_e57_file = pose_data["source_e57_file"]

    data_3d = pose_data["data3d"]
    scan_pose = data_3d["scan_pose"]

    cameras = [
        _parse_camera_pose(
            image_data=image_data,
            pose_id=pose_id,
            source_e57_file=source_e57_file,
        )
        for image_data in pose_data["images"]
    ]

    return ScannerPose(
        pose_id=pose_id,
        source_e57_file=source_e57_file,
        scanner_center_world=np.asarray(
            data_3d["scanner_center_world_m"],
            dtype=np.float64,
        ),
        translation=_parse_translation(scan_pose),
        quaternion_wxyz=_parse_quaternion(scan_pose),
        rotation_matrix=np.asarray(
            scan_pose["rotation_matrix_3x3"],
            dtype=np.float64,
        ),
        transform_matrix=np.asarray(
            scan_pose["transform_matrix_4x4"],
            dtype=np.float64,
        ),
        cameras=cameras,
    )


def _parse_camera_pose(
    image_data: dict,
    pose_id: str,
    source_e57_file: str,
) -> CameraPose:
    """Convert one pinhole image entry to CameraPose."""
    camera_pose = image_data["camera_pose"]
    pinhole = image_data["pinhole"]

    return CameraPose(
        pose_id=pose_id,
        source_e57_file=source_e57_file,
        image_file=image_data["file_name"],
        camera_center_world=np.asarray(
            image_data["camera_center_world_m"],
            dtype=np.float64,
        ),
        translation=_parse_translation(camera_pose),
        quaternion_wxyz=_parse_quaternion(camera_pose),
        rotation_matrix=np.asarray(
            camera_pose["rotation_matrix_3x3"],
            dtype=np.float64,
        ),
        transform_matrix=np.asarray(
            camera_pose["transform_matrix_4x4"],
            dtype=np.float64,
        ),
        image_width=pinhole["image_width"],
        image_height=pinhole["image_height"],
        focal_length_m=pinhole["focal_length_m"],
        pixel_width_m=pinhole["pixel_width_m"],
        pixel_height_m=pinhole["pixel_height_m"],
        fx=pinhole["fx_px"],
        fy=pinhole["fy_px"],
        cx=pinhole["principal_point_x_px"],
        cy=pinhole["principal_point_y_px"],
    )


def _parse_translation(
    pose_data: dict,
) -> np.ndarray:
    """Extract XYZ translation from pose data."""
    translation = pose_data["translation_m"]

    return np.array(
        [
            translation["x"],
            translation["y"],
            translation["z"],
        ],
        dtype=np.float64,
    )


def _parse_quaternion(
    pose_data: dict,
) -> np.ndarray:
    """Extract a WXYZ quaternion from pose data."""
    quaternion = pose_data["rotation_quaternion_wxyz"]

    return np.array(
        [
            quaternion["w"],
            quaternion["x"],
            quaternion["y"],
            quaternion["z"],
        ],
        dtype=np.float64,
    )