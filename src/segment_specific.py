from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path    
import numpy as np
import cv2

# parent dir of the current file
parent_dir = Path(__file__).parent.parent

# load the model
model_path = parent_dir / "models/segment_anything/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=model_path)

# create the predictor
predictor = SamPredictor(sam)

# load the image
image_path = parent_dir / "data/images/warehouse.png"
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# set the image for the predictor
predictor.set_image(image)

masks, scores, logits = predictor.predict(
    point_coords=np.array([[860, 650]]),
    point_labels=np.array([1])
)

# visualize the mask
mask = masks[0]
mask_image = np.zeros_like(image)
mask_image[mask] = [255, 0, 0]  # red color

# overlay the mask on the original image
overlay_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

# save the overlay image    
output_path = parent_dir / "data/images/warehouse_mask.png"
cv2.imwrite(str(output_path), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
