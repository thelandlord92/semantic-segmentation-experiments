from mask_projection.mask2cloud.io import load_e57_point_cloud


cloud = load_e57_point_cloud(
    r"C:\Users\bwindapo\polybox\Reality Capture Data\HXE Building\Exports\RTC\260518_Individual Setups\HXE RTC Scan- EXT 001.e57"
)

print(cloud.points)