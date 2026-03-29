# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import laspy
from utils import cloud_to_image
from pathlib import Path    

# parent dir of the current file
parent_dir = Path(__file__).parent.parent.parent

# load the point cloud
pcd_path = parent_dir / "data/point_clouds/itc_building.las"
pcd = laspy.read(str(pcd_path))

# transforming the point cloud to Numpy
pcd_np = np.vstack((pcd.x, pcd.y, pcd.z, (pcd.red/65535*255).astype(int), (pcd.green/65535*255).astype(int), (pcd.blue/65535*255).astype(int))).transpose()

# ortho-Projection
orthoimage = cloud_to_image(pcd_np, 1.5)

# plotting and exporting
fig = plt.figure(figsize=(np.shape(orthoimage)[1]/72, np.shape(orthoimage)[0]/72), dpi=1000)
fig.add_axes([0,0,1,1])
plt.imshow(orthoimage)
plt.axis('off')
plt.savefig(parent_dir / "results/pcd_ortho.jpg")