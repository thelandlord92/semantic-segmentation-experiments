from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path    
import cv2

# parent dir of the current file
parent_dir = Path(__file__).parent.parent

# load the model
model_path = parent_dir / "models/segment_anything/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=model_path)

mask_generator = SamAutomaticMaskGenerator(sam)

# load the image
image_path = parent_dir / "data/images/warehouse.png"

# read the image using OpenCV as the mask generator expects a numpy array
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# generate masks for the entire image
masks = mask_generator.generate(image)