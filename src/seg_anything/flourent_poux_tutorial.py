# The Base libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy
from pathlib import Path

# The Deep Learning libraries
import torch
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
image_path = parent_dir / "data/images/zurich.png"
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# create the mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# generate masks for the entire image
masks = mask_generator.generate(image)

# function to plot the masks
def sam_masks(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    c_mask=[]
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.8)))
        c_mask.append(img)
    return c_mask

# plot the masks
fig = plt.figure(figsize=(np.shape(image)[1]/72, np.shape(image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(image)
color_mask = sam_masks(masks)
plt.axis('off')
plt.savefig(parent_dir / "data/images/zurich_mask.png")