from mask_projection.mask2cloud.io import load_e57_folder
from mask_projection.mask2cloud.visualization import visualize_point_cloud


clouds = load_e57_folder(
    r"C:\Users\bwindapo\polybox\Reality Capture Data\HXE Building\Exports\RTC\260518_Individual Setups",
    max_files=10,
    include_colors=True,
)

for cloud in clouds:
    print(cloud.colors)

visualize_point_cloud(
    clouds,
    point_size=1.0,
)