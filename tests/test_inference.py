"""Unit tests for inference logic."""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.inference import InferenceModel
from app.config import config


class TestInference:
    """Test inference model wrapper."""
    
    def test_model_initialization(self):
        """Test that model initializes with correct state."""
        model = InferenceModel()
        
        assert model.session is None
        assert model.labels == []
        assert model.model_loaded is False
        
    @patch('app.inference.ort.InferenceSession')
    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    def test_model_load_success(self, mock_open, mock_exists, mock_session):
        """Test successful model loading."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock labels file
        mock_labels = "cat\ndog\nbird\n"
        mock_open.return_value.__enter__.return_value.readlines.return_value = mock_labels.split('\n')
        
        # Mock ONNX session
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Load model
        model = InferenceModel()
        model.load()
        
        # Verify
        assert model.model_loaded is True
        assert model.session is not None
        
    @patch('pathlib.Path.exists')
    def test_model_load_missing_file(self, mock_exists):
        """Test model loading with missing model file."""
        mock_exists.return_value = False
        
        model = InferenceModel()
        
        with pytest.raises(FileNotFoundError):
            model.load()
            
    def test_predict_without_loading(self):
        """Test that predict raises error if model not loaded."""
        model = InferenceModel()
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.predict(dummy_input)
            
    @patch('app.inference.ort.InferenceSession')
    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    def test_predict_success(self, mock_open, mock_exists, mock_session):
        """Test successful prediction."""
        # Setup mocks
        mock_exists.return_value = True
        mock_labels = ["class_0", "class_1", "class_2"]
        mock_open.return_value.__enter__.return_value.readlines.return_value = [
            f"{label}\n" for label in mock_labels
        ]
        
        # Mock ONNX session
        mock_session_instance = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session_instance.get_inputs.return_value = [mock_input]
        
        # Create fake output (logits for 3 classes)
        fake_output = np.array([[2.0, 1.0, 0.5]])
        mock_session_instance.run.return_value = [fake_output]
        mock_session.return_value = mock_session_instance
        
        # Load and predict
        model = InferenceModel()
        model.load()
        
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        predictions, inference_time = model.predict(dummy_input)
        
        # Verify
        assert len(predictions) <= config.TOP_K
        assert inference_time >= 0
        assert all('label' in p and 'confidence' in p for p in predictions)
        assert all(0 <= p['confidence'] <= 1 for p in predictions)
        
    def test_postprocess(self):
        """Test post-processing of model outputs."""
        model = InferenceModel()
        # Create dummy labels
        model.labels = [f"class_{i}" for i in range(1000)]
        
        # Create fake logits (1000 classes)
        fake_output = np.random.randn(1, 1000).astype(np.float32)
        # Make first class have highest score
        fake_output[0, 0] = 10.0
        
        predictions = model._postprocess(fake_output)
        
        # Verify
        assert len(predictions) == config.TOP_K
        assert predictions[0]['label'] == 'class_0'
        assert predictions[0]['confidence'] > 0
        # Confidences should sum to less than or equal to 1
        total_conf = sum(p['confidence'] for p in predictions)
        assert total_conf <= 1.0
