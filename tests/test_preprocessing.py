"""Unit tests for image preprocessing."""
import numpy as np
from PIL import Image
import pytest

from app.preprocessing import preprocess_image, validate_image
from app.config import config


class TestPreprocessing:
    """Test image preprocessing functions."""
    
    def test_preprocess_image_shape(self):
        """Test that preprocessed image has correct shape."""
        # Create a dummy RGB image
        image = Image.new('RGB', (512, 512), color='red')
        
        # Preprocess
        result = preprocess_image(image)
        
        # Check shape
        assert result.shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        
    def test_preprocess_image_type(self):
        """Test that preprocessed image is float32."""
        image = Image.new('RGB', (256, 256), color='blue')
        result = preprocess_image(image)
        
        assert result.dtype == np.float32
        
    def test_preprocess_grayscale_image(self):
        """Test preprocessing of grayscale image (should convert to RGB)."""
        image = Image.new('L', (128, 128), color=128)
        result = preprocess_image(image)
        
        # Should still output 3 channels
        assert result.shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        
    def test_preprocess_rgba_image(self):
        """Test preprocessing of RGBA image (should convert to RGB)."""
        image = Image.new('RGBA', (128, 128), color=(255, 0, 0, 128))
        result = preprocess_image(image)
        
        # Should still output 3 channels
        assert result.shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        
    def test_preprocess_normalization(self):
        """Test that normalization produces reasonable values."""
        # Create a white image
        image = Image.new('RGB', (224, 224), color='white')
        result = preprocess_image(image)
        
        # After ImageNet normalization, values should be centered around 0
        # White pixels (255, 255, 255) -> (1.0, 1.0, 1.0) -> normalized
        # The normalized values will be positive since white > mean
        assert result.min() > -5.0  # Reasonable bounds
        assert result.max() < 5.0
        
    def test_validate_image_valid(self):
        """Test validation of a valid image."""
        image = Image.new('RGB', (100, 100))
        is_valid, msg = validate_image(image)
        
        assert is_valid is True
        assert msg == ""
        
    def test_validate_image_invalid_dimensions(self):
        """Test validation of image with invalid dimensions."""
        # PIL doesn't allow 0-sized images, so we mock this scenario
        # by creating an image and checking validation logic
        image = Image.new('RGB', (0, 0))
        is_valid, _ = validate_image(image)
        
        # Should pass - 0x0 is invalid
        assert is_valid is False
        
    def test_preprocess_different_sizes(self):
        """Test preprocessing images of various sizes."""
        sizes = [(64, 64), (128, 256), (512, 128), (1024, 1024)]
        
        for size in sizes:
            image = Image.new('RGB', size)
            result = preprocess_image(image)
            
            # All should be resized to same output
            assert result.shape == (1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
