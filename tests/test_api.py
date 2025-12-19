"""API-level integration tests."""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from unittest.mock import patch, MagicMock
import numpy as np

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock the model to avoid loading actual ONNX model in tests."""
    with patch('app.main.model') as mock:
        mock.model_loaded = True
        mock.predict.return_value = (
            [
                {"label": "tabby_cat", "confidence": 0.87},
                {"label": "tiger_cat", "confidence": 0.08},
                {"label": "Egyptian_cat", "confidence": 0.03}
            ],
            11.2
        )
        yield mock


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check(self, client, mock_model):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "ok"
        

class TestInferEndpoint:
    """Test /infer endpoint."""
    
    def create_test_image(self, mode='RGB', size=(224, 224), color='red'):
        """Helper to create test image bytes."""
        image = Image.new(mode, size, color=color)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
    
    def test_infer_success(self, client, mock_model):
        """Test successful inference request."""
        img_bytes = self.create_test_image()
        
        response = client.post(
            "/infer",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "predictions" in data
        assert "inference_time_ms" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0
        
        # Verify prediction structure
        for pred in data["predictions"]:
            assert "label" in pred
            assert "confidence" in pred
            assert isinstance(pred["confidence"], (int, float))
            
    def test_infer_invalid_file_type(self, client, mock_model):
        """Test inference with invalid file type."""
        # Create a text file instead of image
        text_file = io.BytesIO(b"This is not an image")
        
        response = client.post(
            "/infer",
            files={"file": ("test.txt", text_file, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
        
    def test_infer_no_file(self, client, mock_model):
        """Test inference without providing file."""
        response = client.post("/infer")
        
        assert response.status_code == 422  # Validation error
        
    def test_infer_grayscale_image(self, client, mock_model):
        """Test inference with grayscale image."""
        img_bytes = self.create_test_image(mode='L')
        
        response = client.post(
            "/infer",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        assert response.status_code == 200
        
    def test_infer_different_sizes(self, client, mock_model):
        """Test inference with images of different sizes."""
        sizes = [(64, 64), (512, 512), (100, 200)]
        
        for size in sizes:
            img_bytes = self.create_test_image(size=size)
            
            response = client.post(
                "/infer",
                files={"file": ("test.png", img_bytes, "image/png")}
            )
            
            assert response.status_code == 200
            
    def test_infer_response_format(self, client, mock_model):
        """Test that inference response matches expected format."""
        img_bytes = self.create_test_image()
        
        response = client.post(
            "/infer",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        
        data = response.json()
        
        # Verify exact format matches specification
        assert "predictions" in data
        assert "inference_time_ms" in data
        assert isinstance(data["predictions"], list)
        
        for pred in data["predictions"]:
            assert set(pred.keys()) == {"label", "confidence"}


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
