"""Functions for reading E57 point-cloud data."""

from pathlib import Path

import numpy as np
import pye57
from numpy.typing import NDArray

from ..models import PointCloudData


def load_e57_point_cloud(
    file_path: str | Path,
    scan_index: int = 0,
    point_id_offset: int = 0,
    apply_pose: bool = True,
) -> PointCloudData:
    """Load a point cloud from an E57 scan.

    The original E57 point indices are retained as point identifiers.
    Invalid and non-finite points are removed without changing the
    identifiers of the remaining points.

    Args:
        file_path: Path to the E57 file.
        scan_index: Index of the scan within the E57 file.
        point_id_offset: Offset added to the original point indices.
        apply_pose: Whether to transform points into world coordinates.

    Returns:
        Point-cloud data and associated scan metadata.

    Raises:
        FileNotFoundError: If the E57 file does not exist.
        ValueError: If the file is not an E57 file or Cartesian
            coordinates are unavailable.
        IndexError: If the requested scan index does not exist.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"E57 file not found: {path}")

    if path.suffix.lower() != ".e57":
        raise ValueError(f"Expected an E57 file, received: {path.suffix}")

    e57 = pye57.E57(str(path))

    if not 0 <= scan_index < e57.scan_count:
        raise IndexError(
            f"Scan index {scan_index} is outside the valid range "
            f"0 to {e57.scan_count - 1}."
        )

    data = e57.read_scan_raw(scan_index)

    coordinate_fields = (
        "cartesianX",
        "cartesianY",
        "cartesianZ",
    )

    missing_fields = [
        field for field in coordinate_fields if field not in data
    ]

    if missing_fields:
        raise ValueError(
            "E57 scan does not contain Cartesian coordinates. "
            f"Missing fields: {missing_fields}"
        )

    points = np.column_stack(
        (
            data["cartesianX"],
            data["cartesianY"],
            data["cartesianZ"],
        )
    ).astype(np.float64, copy=False)

    raw_point_count = points.shape[0]

    point_ids = np.arange(
        point_id_offset,
        point_id_offset + raw_point_count,
        dtype=np.int64,
    )

    valid_mask = np.all(np.isfinite(points), axis=1)

    if "cartesianInvalidState" in data:
        invalid_state = np.asarray(data["cartesianInvalidState"])
        valid_mask &= invalid_state == 0

    points = points[valid_mask]
    point_ids = point_ids[valid_mask]

    header = e57.get_header(scan_index)

    rotation = np.asarray(
        header.rotation_matrix,
        dtype=np.float64,
    )
    translation = np.asarray(
        header.translation,
        dtype=np.float64,
    )

    local_to_world = np.eye(4, dtype=np.float64)
    local_to_world[:3, :3] = rotation
    local_to_world[:3, 3] = translation

    if apply_pose:
        points = points @ rotation.T + translation

    intensity = _extract_optional_field(
        data,
        "intensity",
        valid_mask,
        np.float64,
    )

    row_indices = _extract_optional_field(
        data,
        "rowIndex",
        valid_mask,
        np.int64,
    )

    column_indices = _extract_optional_field(
        data,
        "columnIndex",
        valid_mask,
        np.int64,
    )

    colors = _extract_colors(data, valid_mask)

    return PointCloudData(
        points=points,
        point_ids=point_ids,
        source_path=path,
        scan_index=scan_index,
        raw_point_count=raw_point_count,
        local_to_world=local_to_world,
        intensity=intensity,
        colors=colors,
        row_indices=row_indices,
        column_indices=column_indices,
    )


def _extract_optional_field(
    data: dict,
    field_name: str,
    valid_mask: NDArray[np.bool_],
    dtype: type,
) -> NDArray | None:
    """Extract an optional E57 field using the valid-point mask."""
    if field_name not in data:
        return None

    values = np.asarray(data[field_name])[valid_mask]
    return values.astype(dtype, copy=False)


def _extract_colors(
    data: dict,
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float64] | None:
    """Extract RGB values from E57 data when available."""
    color_fields = (
        "colorRed",
        "colorGreen",
        "colorBlue",
    )

    if not all(field in data for field in color_fields):
        return None

    colors = np.column_stack(
        tuple(data[field] for field in color_fields)
    )

    return colors[valid_mask].astype(np.float64, copy=False)