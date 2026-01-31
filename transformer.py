import httpx
from io import BytesIO
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with httpx.stream("GET", url) as response:
    image = Image.open(BytesIO(response.read()))
# Check for cats and remote controls
text_labels = [["a cat", "a remote control"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[(image.height, image.width)]
)
# Retrieve the first image result
# result = results[0]
# for box, score, text_label in zip(result["boxes"], result["scores"], result["text_labels"]):
#     box = [round(x, 2) for x in box.tolist()]
#     print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")

# Assuming a single image, results[0] is a dict with keys 'boxes', 'labels', 'scores'
dino_boxes = results[0]["boxes"]      # shape: [num_objects, 4], in xyxy format
dino_labels = results[0]["labels"]    # list of labels
dino_scores = results[0]["scores"]    # list of confidence scores

model_name = "sushmanth/sam_hq_vit_b"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

import torch
import numpy as np
import matplotlib.pyplot as plt

# Assuming:
# image -> PIL.Image
# dino_boxes, dino_labels, dino_scores already defined
# processor and model loaded from Hugging Face

device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Normalize boxes to [0,1] for SAM
img_height, img_width = image.height, image.width
boxes_norm = []
for box in dino_boxes:
    x0, y0, x1, y1 = box
    boxes_norm.append([x0 / img_width, y0 / img_height, x1 / img_width, y1 / img_height])

boxes_tensor = torch.tensor(boxes_norm, dtype=torch.float32).to(device)

# Step 2: Prepare SAM inputs
inputs = processor(
    images=image,
    boxes=boxes_tensor,
    return_tensors="pt"
)
for k in inputs:
    inputs[k] = inputs[k].to(device)

# Step 3: Run SAM
with torch.no_grad():
    outputs = model(**inputs)

# raw masks
masks_raw = torch.sigmoid(outputs.pred_masks)  # [1, 1, 3, H, W]

# remove batch & box dims
masks = masks_raw[0, 0]  # now shape = [3, H, W]

# Combine the 3 channels into 1 mask (take max over channels)
masks_2d = (masks.max(dim=0).values > 0.5).cpu().numpy()  # shape = [H, W]


print(masks_2d.shape)  # should now be [num_boxes, H, W]

image_np = np.array(image)

plt.figure(figsize=(10,10))
plt.imshow(image_np)
plt.imshow(masks_2d, alpha=0.5, cmap="Reds")  # single-color overlay
plt.axis("off")
plt.show()




