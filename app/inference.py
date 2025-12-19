"""ONNX model inference wrapper."""
import time
from typing import List, Dict, Tuple
import numpy as np
import onnxruntime as ort
from pathlib import Path

from app.config import config


class InferenceModel:
    """
    Wrapper for ONNX Runtime inference session.
    
    This class handles:
    - Loading the ONNX model and labels at startup
    - Running inference in a thread-safe manner
    - Post-processing outputs to get top-k predictions
    """
    
    def __init__(self):
        """Initialize the inference model (loads model and labels)."""
        self.session = None
        self.labels = []
        self.model_loaded = False
        
    def load(self) -> None:
        """
        Load the ONNX model and labels from disk.
        
        Raises:
            FileNotFoundError: If model or labels file not found
            Exception: If model loading fails
        """
        model_path = Path(config.MODEL_PATH)
        labels_path = Path(config.LABELS_PATH)
        
        # Validate paths
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        # Configure ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config.NUM_THREADS
        sess_options.inter_op_num_threads = config.NUM_THREADS
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        self.model_loaded = True
        
    def predict(self, image_array: np.ndarray) -> Tuple[List[Dict[str, float]], float]:
        """
        Run inference on preprocessed image.
        
        Args:
            image_array: Preprocessed image array of shape (1, 3, 224, 224)
            
        Returns:
            Tuple of (predictions, inference_time_ms) where predictions is a list
            of dicts with 'label' and 'confidence' keys
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.model_loaded or self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Get input name from the model
        input_name = self.session.get_inputs()[0].name
        
        # Run inference and measure time
        start_time = time.perf_counter()
        outputs = self.session.run(None, {input_name: image_array})
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Post-process outputs
        predictions = self._postprocess(outputs[0])
        
        return predictions, inference_time
    
    def _postprocess(self, output: np.ndarray) -> List[Dict[str, float]]:
        """
        Post-process model output to get top-k predictions.
        
        Args:
            output: Raw model output of shape (1, 1000)
            
        Returns:
            List of top-k predictions with label and confidence
        """
        # Apply softmax to get probabilities
        logits = output[0]  # Remove batch dimension
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[-config.TOP_K:][::-1]
        
        # Build predictions list
        predictions = []
        for idx in top_k_indices:
            predictions.append({
                "label": self.labels[idx] if idx < len(self.labels) else f"class_{idx}",
                "confidence": round(float(probabilities[idx]), 4)
            })
        
        return predictions


# Global model instance (singleton pattern)
model = InferenceModel()
