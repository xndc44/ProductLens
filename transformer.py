import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from datasets import load_dataset


dataset = load_dataset("productLensData/images")
images = dataset['train']['image'][0]

image_processor = AutoImageProcessor.from_pretrained("./deformable-detr")
model = DeformableDetrForObjectDetection.from_pretrained("./deformable-detr")

# configuration = DeformableDetrConfig() # change configuration.values to change architecture
# model = DeformableDetrModel(configuration)
# configuration = model.config

inputs = image_processor(images, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([images.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )


