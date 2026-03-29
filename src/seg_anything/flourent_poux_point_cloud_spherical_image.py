# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import laspy
from utils import generate_spherical_image
from pathlib import Path    

# parameters for the spherical image
resolution = 500
camera_center_coords = [-69, -92, 2]

# parent dir of the current file
parent_dir = Path(__file__).parent.parent.parent

# load the point cloud
pcd_path = parent_dir / "data/point_clouds/fribourg_building_b.las"
pcd = laspy.read(str(pcd_path))

# transform the point cloud to a Numpy array
coords = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()

# gather the colors
r=(pcd.red/65535*255).astype(int)
g=(pcd.green/65535*255).astype(int)
b=(pcd.blue/65535*255).astype(int)
colors = np.vstack((r,g,b)).transpose()

# function execution
image, mapping = generate_spherical_image(camera_center_coords, coords, colors, resolution)

print(image[0])

# plotting with matplotlib
fig = plt.figure(figsize=(np.shape(image)[1]/72, np.shape(image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(image)
plt.axis('off')

# saving to the disk
plt.savefig(parent_dir / "results/fribourg_building_b.jpg")