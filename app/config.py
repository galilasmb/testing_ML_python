"""Configuration management for the inference service."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Application configuration from environment variables."""
    
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/resnet50.onnx")
    LABELS_PATH: str = os.getenv("LABELS_PATH", "models/imagenet_classes.txt")
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    NUM_THREADS: int = int(os.getenv("NUM_THREADS", "4"))
    
    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224


config = Config()
