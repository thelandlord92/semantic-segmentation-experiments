# The Base libraries
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_spherical_image, sam_masks, color_point_cloud
from utils import masks_to_color_image, export_point_cloud
import laspy

# The Deep Learning libraries
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# The Open3D library
import open3d as o3d

######## code for generating the spherical image from the point cloud ########

# parameters for the spherical image
resolution = 500
camera_center_coords = [189, 60, 2]

# parent dir of the current file
parent_dir = Path(__file__).parent.parent.parent

# load the point cloud
pcd_path = parent_dir / "data/point_clouds/itc_building.las"
pcd = laspy.read(str(pcd_path))

# transform the point cloud to a Numpy array
coords = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()

# gather the colors
r=(pcd.red/65535*255).astype(int)
g=(pcd.green/65535*255).astype(int)
b=(pcd.blue/65535*255).astype(int)
colors = np.vstack((r,g,b)).transpose()

# function execution
spherical_image, mapping = generate_spherical_image(camera_center_coords, coords, colors, resolution)

print("spherical image generated \n")

################### code for segmenting the spherical image ####################

# load the segment anything model
model_path = parent_dir / "models/segment_anything/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=model_path)

# cast the model to CPU
sam.to("cpu")

# load the image
image = spherical_image.astype(np.uint8)

print("spherical image loaded \n")

# create the mask generator
t0 = time.time()
mask_generator = SamAutomaticMaskGenerator(sam)
t1 = time.time()

# generate masks for the entire image
masks = mask_generator.generate(image)

print("masks generated \n")

# plot the masks
fig = plt.figure(figsize=(np.shape(image)[1]/72, np.shape(image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(image)
color_mask = sam_masks(masks)
plt.axis('off')
plt.savefig(parent_dir / "results/itc_mask.jpg")

print("mask plotted and saved \n")


###### code to color and save the point cloud with the masks ########

# convert the masks to a color image
color_mask = masks_to_color_image(masks, image.shape)

print("color mask generated \n")

modified_point_cloud = color_point_cloud(color_mask, coords, mapping)

print("point cloud colored \n")
print(type(modified_point_cloud))
print(modified_point_cloud.shape)


###### code to visualize the colored point cloud ########
view_point_cloud = False
if view_point_cloud:
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(modified_point_cloud[:, :3])
    pcd_colored.colors = o3d.utility.Vector3dVector(modified_point_cloud[:, 3:6] / 255.0)
    o3d.visualization.draw_geometries([pcd_colored])


###### code to export the colored point cloud ########
cloud_path = parent_dir / "results/itc_colored_point_cloud.las"
export_point_cloud(cloud_path, modified_point_cloud)   