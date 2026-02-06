import numpy as np
from PIL import Image
from .sam import segment_image
from .dino import get_embedding
from typing import List, Dict

def run_vision_pipeline(image: Image.Image) -> List[Dict]:
    """
    Detects objects using SAM and generates embeddings with DINOv2.

    Args:
        image (Image.Image): Input PIL image

    Returns:
        List[Dict]: List of objects with bbox and embedding
    """
    image_np = np.array(image)
    masks = segment_image(image_np)
    objects = []

    for i, m in enumerate(masks):
        x, y, w, h = m["bbox"]
        crop = image.crop((x, y, x + w, y + h))
        embedding = get_embedding(crop)

        objects.append({
            "id": f"obj_{i}",
            "bbox": m["bbox"],
            "embedding": embedding.tolist()
        })

    return objects
