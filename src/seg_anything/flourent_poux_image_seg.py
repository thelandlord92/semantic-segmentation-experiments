# The Base libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from utils import sam_masks

# The Deep Learning libraries
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

# parent dir of the current file
parent_dir = Path(__file__).parent.parent.parent

# load the model
model_path = parent_dir / "models/segment_anything/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=model_path)

# cast the model to CPU
sam.to("cpu")

# load the image
image_path = parent_dir / "data/images/warehouse.png"
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# create the mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# generate masks for the entire image
masks = mask_generator.generate(image)

# plot the masks
fig = plt.figure(figsize=(np.shape(image)[1]/72, np.shape(image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(image)
color_mask = sam_masks(masks)
plt.axis('off')
plt.savefig(parent_dir / "results/warehouse_mask.jpg")
