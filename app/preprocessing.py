"""Image preprocessing utilities for ResNet-50 inference."""
import numpy as np
from PIL import Image
from typing import Tuple

from app.config import config


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an image for ResNet-50 ONNX model inference.
    
    The ResNet-50 model expects:
    - Input shape: (1, 3, 224, 224) - NCHW format
    - RGB channels
    - Normalized with ImageNet mean/std
    
    Args:
        image: PIL Image object
        
    Returns:
        numpy array of shape (1, 3, 224, 224) ready for inference
    """
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224
    image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(config.IMAGENET_STD, dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format (Height, Width, Channels -> Channels, Height, Width)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate that an image is suitable for processing.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if image has valid dimensions
        width, height = image.size
        if width < 1 or height < 1:
            return False, "Image has invalid dimensions"
        
        # Check if image format is supported
        if image.mode not in ['RGB', 'RGBA', 'L', 'P']:
            return False, f"Unsupported image mode: {image.mode}"
        
        return True, ""
    except Exception as e:
        return False, f"Image validation error: {str(e)}"
