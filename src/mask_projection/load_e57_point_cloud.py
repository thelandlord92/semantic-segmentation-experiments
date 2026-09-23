from pathlib import Path

from mask_projection.mask2cloud.io import (
    load_e57_folder,
    load_pose_data,
)
from mask_projection.mask2cloud.visualization import (
    create_point_cloud_geometries,
    create_scanner_pose_geometries,
    visualize_scene,
)

# -----------------------------
# DIRECTORIES
# -----------------------------
point_cloud_dir = Path(
    r"C:\Users\bwindapo\polybox\Reality Capture Data\HXE Building\Exports\RTC\260518_Individual Setups"
)

parent_dir = Path(__file__).parent.parent.parent
pose_dir = parent_dir / "data" / "pose_data" / "hxe" / "pose_data.json"

# -----------------------------
# Load point clouds from an E57
# -----------------------------
clouds = load_e57_folder(
    point_cloud_dir,
    max_files=5,
    include_colors=True,
)

for cloud in clouds:
    print(cloud.source_path)

# -----------------------------
# # Load the pose data from the JSON file
# -----------------------------
poses = load_pose_data(pose_dir)


#-----------------------------
# Create geometries for visualization
#-----------------------------
point_cloud_geometries = (
    create_point_cloud_geometries(
        clouds,
        use_colors=True,
    )
)

scanner_geometries = (
    create_scanner_pose_geometries(
        poses,
        point_clouds=clouds,
        marker_radius=0.15,
        coordinate_frame_size=0.75,
    )
)

geometries = (
    point_cloud_geometries
    + scanner_geometries
)


visualize_scene(
    geometries,
    point_size=1.0,
    show_world_frame=False,
    world_frame_size=2.0,
)
