"""
Pydantic schemas for batch inference endpoints.
"""
from typing import List
from pydantic import BaseModel, ConfigDict, Field


class BatchPrediction(BaseModel):
    """Single image prediction in batch response."""
    
    filename: str = Field(..., description="Original filename of the image")
    predictions: List[dict] = Field(..., description="Top-k predictions for this image")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    error: str | None = Field(None, description="Error message if processing failed")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "cat.jpg",
                "predictions": [
                    {"label": "tabby_cat", "confidence": 0.87},
                    {"label": "tiger_cat", "confidence": 0.08}
                ],
                "inference_time_ms": 12.5,
                "error": None
            }
        }
    )


class BatchInferenceResponse(BaseModel):
    """Response for batch inference endpoint."""
    
    total_images: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successfully processed images")
    failed: int = Field(..., description="Number of failed images")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    results: List[BatchPrediction] = Field(..., description="Individual results for each image")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_images": 3,
                "successful": 2,
                "failed": 1,
                "total_time_ms": 45.3,
                "results": [
                    {
                        "filename": "cat.jpg",
                        "predictions": [{"label": "tabby_cat", "confidence": 0.87}],
                        "inference_time_ms": 12.5,
                        "error": None
                    },
                    {
                        "filename": "dog.jpg",
                        "predictions": [{"label": "golden_retriever", "confidence": 0.92}],
                        "inference_time_ms": 11.8,
                        "error": None
                    },
                    {
                        "filename": "invalid.txt",
                        "predictions": [],
                        "inference_time_ms": 0.0,
                        "error": "Invalid image format"
                    }
                ]
            }
        }
    )


class MetricsResponse(BaseModel):
    """Response for metrics endpoint."""
    
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    success_rate: float = Field(..., description="Success rate percentage")
    average_latency_ms: float = Field(..., description="Average inference latency in ms")
    p95_latency_ms: float = Field(..., description="95th percentile latency in ms")
    p99_latency_ms: float = Field(..., description="99th percentile latency in ms")
    requests_per_second: float = Field(..., description="Requests per second")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "uptime_seconds": 3600.5,
                "total_requests": 1250,
                "successful_requests": 1230,
                "failed_requests": 20,
                "success_rate": 98.4,
                "average_latency_ms": 15.3,
                "p95_latency_ms": 28.5,
                "p99_latency_ms": 42.1,
                "requests_per_second": 0.35
            }
        }
    )
