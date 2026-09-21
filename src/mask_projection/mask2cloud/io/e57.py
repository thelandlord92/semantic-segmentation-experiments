"""Functions for reading E57 point-cloud data."""

from pathlib import Path
from typing import Any

import numpy as np
import pye57
import warnings

from ..models import PointCloudData


def load_e57_point_cloud(
    file_path: str | Path,
    point_id_offset: int = 0,
    apply_pose: bool = True,
    include_intensity: bool = False,
    include_colors: bool = False,
    include_row_column: bool = False,
) -> PointCloudData:
    """Load point-cloud data from a single-scan E57 file.

    Args:
        file_path: Path to the E57 file.
        point_id_offset: Starting identifier assigned to loaded points.
        apply_pose: Whether to transform points into world coordinates.
        include_intensity: Whether to load intensity values.
        include_colors: Whether to load RGB values.
        include_row_column: Whether to load E57 row and column indices.

    Returns:
        Point-cloud data and associated scan metadata.

    Raises:
        FileNotFoundError: If the E57 file does not exist.
        ValueError: If the path is not an E57 file or the E57 does not
            contain exactly one scan.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"E57 file not found: {path}")

    if path.suffix.lower() != ".e57":
        raise ValueError(
            f"Expected an E57 file, received: {path.suffix}"
        )

    scan_index = 0

    with pye57.E57(str(path)) as e57:
        if e57.scan_count != 1:
            raise ValueError(
                f"Expected one scan in {path.name}, "
                f"but found {e57.scan_count}."
            )

        header = e57.get_header(scan_index)

        data = e57.read_scan(
            scan_index,
            intensity=include_intensity,
            colors=include_colors,
            row_column=include_row_column,
            transform=apply_pose,
            ignore_missing_fields=True,
        )

        raw_point_count = header.point_count
        local_to_world = _extract_local_to_world(header)

    points = np.column_stack(
        (
            data["cartesianX"],
            data["cartesianY"],
            data["cartesianZ"],
        )
    ).astype(np.float64, copy=False)

    point_count = points.shape[0]

    point_ids = np.arange(
        point_id_offset,
        point_id_offset + point_count,
        dtype=np.int64,
    )

    intensity = _extract_optional_field(
        data=data,
        field_name="intensity",
        dtype=np.float64,
    )

    row_indices = _extract_optional_field(
        data=data,
        field_name="rowIndex",
        dtype=np.int64,
    )

    column_indices = _extract_optional_field(
        data=data,
        field_name="columnIndex",
        dtype=np.int64,
    )

    colors = _extract_colors(data)

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


def load_e57_folder(
    folder_path: str | Path,
    apply_pose: bool = True,
    include_intensity: bool = False,
    include_colors: bool = False,
    include_row_column: bool = False,
    skip_errors: bool = True,
    max_files: int | None = None,
) -> list[PointCloudData]:
    """Load single-scan E57 files from a folder.

    Args:
        folder_path: Path containing E57 files.
        apply_pose: Whether to transform points into world coordinates.
        include_intensity: Whether to load intensity values.
        include_colors: Whether to load RGB values.
        include_row_column: Whether to load E57 row and column indices.
        skip_errors: Whether to skip E57 files that cannot be read.
        max_files: Maximum number of E57 files to attempt to load.
            If ``None``, all files are considered.

    Returns:
        Point-cloud data for successfully loaded E57 files.

    Raises:
        NotADirectoryError: If the folder does not exist.
        FileNotFoundError: If no E57 files are found.
        ValueError: If ``max_files`` is less than 1.
        RuntimeError: If a file cannot be read and ``skip_errors`` is False.
    """
    folder = Path(folder_path).expanduser().resolve()

    if not folder.is_dir():
        raise NotADirectoryError(
            f"E57 folder not found: {folder}"
        )

    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be at least 1 or None.")

    e57_files = sorted(
        (
            file_path
            for file_path in folder.iterdir()
            if file_path.is_file()
            and file_path.suffix.lower() == ".e57"
        ),
        key=lambda file_path: file_path.name.lower(),
    )

    if not e57_files:
        raise FileNotFoundError(
            f"No E57 files found in: {folder}"
        )

    if max_files is not None:
        e57_files = e57_files[:max_files]

    point_clouds: list[PointCloudData] = []
    point_id_offset = 0

    for file_path in e57_files:
        try:
            point_cloud = load_e57_point_cloud(
                file_path=file_path,
                point_id_offset=point_id_offset,
                apply_pose=apply_pose,
                include_intensity=include_intensity,
                include_colors=include_colors,
                include_row_column=include_row_column,
            )

        except Exception as exc:
            if not skip_errors:
                raise RuntimeError(
                    f"Failed to load E57 file: {file_path.name}"
                ) from exc

            warnings.warn(
                f"Skipping E57 file '{file_path.name}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        point_clouds.append(point_cloud)
        point_id_offset += point_cloud.points.shape[0]

    return point_clouds


def _extract_local_to_world(header: Any) -> np.ndarray:
    """Extract the E57 scan pose as a 4 x 4 transformation matrix.

    Args:
        header: E57 scan header.

    Returns:
        Local-to-world homogeneous transformation matrix.
    """
    local_to_world = np.eye(4, dtype=np.float64)

    if header.has_pose():
        local_to_world[:3, :3] = np.asarray(
            header.rotation_matrix,
            dtype=np.float64,
        )
        local_to_world[:3, 3] = np.asarray(
            header.translation,
            dtype=np.float64,
        )

    return local_to_world


def _extract_optional_field(
    data: dict[str, Any],
    field_name: str,
    dtype: type,
) -> np.ndarray | None:
    """Extract an optional E57 field.

    Args:
        data: Data dictionary returned by ``pye57.E57.read_scan``.
        field_name: Name of the field to extract.
        dtype: NumPy data type for the returned array.

    Returns:
        Requested array, or ``None`` when the field is unavailable.
    """
    if field_name not in data:
        return None

    return np.asarray(
        data[field_name],
        dtype=dtype,
    )


def _extract_colors(
    data: dict[str, Any],
) -> np.ndarray | None:
    """Extract RGB values when available.

    Args:
        data: Data dictionary returned by ``pye57.E57.read_scan``.

    Returns:
        An ``(N, 3)`` RGB array, or ``None`` if color data is unavailable.
    """
    color_fields = (
        "colorRed",
        "colorGreen",
        "colorBlue",
    )

    if not all(field in data for field in color_fields):
        return None

    return np.column_stack(
        tuple(data[field] for field in color_fields)
    ).astype(np.float64, copy=False)
