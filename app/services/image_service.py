import io
import os
import uuid
from typing import List
from PIL import Image
from app.config import settings

def stitch_images(image_bytes_list: List[bytes]) -> Image.Image:
    """
    Stitches up to 4 images into a 2x2 grid.
    """
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
    
    if not images:
        raise ValueError("No images provided")
    
    if len(images) == 1:
        return images[0]
    
    # Limit to 4 images
    images = images[:4]
    
    # Resize to match the first image
    w, h = images[0].size
    resized_images = [img.resize((w, h)) for img in images]
    
    # Create grid canvas
    grid_w = w * 2
    grid_h = h * 2
    grid_img = Image.new('RGB', (grid_w, grid_h))
    
    # Paste images
    grid_img.paste(resized_images[0], (0, 0))
    if len(resized_images) > 1:
        grid_img.paste(resized_images[1], (w, 0))
    if len(resized_images) > 2:
        grid_img.paste(resized_images[2], (0, h))
    if len(resized_images) > 3:
        grid_img.paste(resized_images[3], (w, h))
        
    return grid_img

def save_image_locally(image: Image.Image, prefix: str) -> str:
    """Saves image to disk and returns the absolute path."""
    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(settings.IMAGES_DIR, filename)
    image.save(path, "JPEG", quality=85)
    return path
