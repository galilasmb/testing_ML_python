"""Pydantic models for API request/response schemas."""
from pydantic import BaseModel, Field, ConfigDict
from typing import List


class Prediction(BaseModel):
    """Single prediction result."""
    label: str = Field(..., description="Class label name", json_schema_extra={"example": "tabby_cat"})
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1", json_schema_extra={"example": 0.87})


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {"label": "tabby_cat", "confidence": 0.87},
                    {"label": "tiger_cat", "confidence": 0.08},
                    {"label": "Egyptian_cat", "confidence": 0.03}
                ],
                "inference_time_ms": 47.3
            }
        }
    )
    
    predictions: List[Prediction] = Field(
        ..., 
        description="List of top-k predictions"
    )
    inference_time_ms: float = Field(
        ..., 
        description="Inference time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True
            }
        }
    )
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")


class ServiceInfo(BaseModel):
    """Response model for root endpoint."""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    endpoints: dict = Field(..., description="Available endpoints")


class ErrorResponse(BaseModel):
    """Error response model."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Invalid file type. Expected image, got: text/plain"
            }
        }
    )
    
    detail: str = Field(..., description="Error message")
