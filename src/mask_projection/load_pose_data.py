from pathlib import Path

from mask_projection.mask2cloud.io import load_pose_data

# -----------------------------
# POSE DIRECTORY
# -----------------------------
parent_dir = Path(__file__).parent.parent.parent
pose_dir = parent_dir / "data" / "pose_data" / "hxe" / "pose_data.json"

# Load the pose data from the JSON file
poses = load_pose_data(pose_dir)

print(poses[0].source_e57_file)
print(poses[0].scanner_center_world)

camera = poses[0].cameras[0]

print(camera.image_file)
print(camera.intrinsic_matrix)
print(camera.world_to_camera)

print(len(poses))