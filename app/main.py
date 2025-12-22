"""FastAPI application for image classification inference."""
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Dict, List, Any
import logging
from contextlib import asynccontextmanager
import time

from app.inference import model
from app.preprocessing import preprocess_image, validate_image
from app.schemas import InferenceResponse, HealthResponse, ServiceInfo, ErrorResponse
from app.batch_schemas import BatchInferenceResponse, BatchPrediction, MetricsResponse
from app.metrics import metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan - load model on startup, cleanup on shutdown.
    
    This is the modern FastAPI way to handle startup/shutdown events.
    """
    # Startup
    try:
        logger.info("Loading ONNX model and labels...")
        model.load()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Image Inference Service",
    description="""
    ## Image Classification API using ResNet-50 ONNX Model
    
    This service provides real-time image classification using a pretrained ResNet-50 model.
    
    ### Features:
    - Fast Inference: ~15-50ms per image on CPU
    - ImageNet Classes Recognizes 1000 different object categories
    - Any Image Format: Supports JPEG, PNG, and other common formats
    - Automatic Preprocessing: Images are automatically resized and normalized
    
    ### Model Information:
    - Architecture: ResNet-50
    - Framework: ONNX Runtime
    - Dataset: ImageNet (1000 classes)
    - Input Size: 224x224 pixels (automatic resizing)
    
    ### Usage:
    1. Upload an image to the `/infer` endpoint
    2. Receive top-k predictions with confidence scores
    3. Monitor service health via `/health` endpoint

    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "NONE",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and service status endpoints"
        },
        {
            "name": "inference",
            "description": "Image classification inference operations"
        },
        {
            "name": "info",
            "description": "Service information"
        }
    ]
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health Check",
    description="Check if the service is running and the model is loaded",
    responses={
        200: {
            "description": "Service is healthy and model is loaded",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "model_loaded": True
                    }
                }
            }
        }
    }
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status and model readiness. This endpoint is useful
    for container orchestration systems to determine if the service is ready
    to accept requests.
    
    Returns:
        HealthResponse: Status and model loading state
    """
    return HealthResponse(
        status="ok",
        model_loaded=model.model_loaded
    )


@app.post(
    "/infer",
    response_model=InferenceResponse,
    tags=["inference"],
    summary="Classify Image",
    description="""
    Classify an uploaded image using the ResNet-50 model.
    
    The endpoint accepts an image file and returns the top-k predictions with confidence scores.
    
    Supported formats: JPEG, PNG, GIF, BMP, and other common image formats
    
    Processing:
    - Image is automatically resized to 224x224 pixels
    - ImageNet normalization is applied
    - Inference runs on CPU using ONNX Runtime
    
    Returns: Top-k class predictions (default k=3) with confidence scores
    """,
    responses={
        200: {
            "description": "Successful inference",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {"label": "tabby_cat", "confidence": 0.87},
                            {"label": "tiger_cat", "confidence": 0.08},
                            {"label": "Egyptian_cat", "confidence": 0.03}
                        ],
                        "inference_time_ms": 47.3
                    }
                }
            }
        },
        400: {
            "description": "Invalid input (wrong file type or corrupted image)",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid file type. Expected image, got: text/plain"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error during inference",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error during inference: Model not loaded"
                    }
                }
            }
        }
    }
)
async def infer(
    file: UploadFile = File(
        ...,
        description="Image file to classify"
    )
) -> InferenceResponse:
    """
    Image classification inference endpoint.
    
    This endpoint is intentionally async even though the inference itself is sync.
    The async wrapper allows FastAPI to handle multiple requests concurrently
    while the actual CPU-bound inference happens synchronously.
    
    Design decision: We run inference synchronously because ONNX Runtime with
    CPU provider is CPU-bound and doesn't benefit from async/await. However,
    the FastAPI endpoint is async so it doesn't block the event loop during
    file upload/download operations.
    
    Args:
        file: Uploaded image file (multipart/form-data)
        
    Returns:
        JSON response with predictions and inference time
        
    Raises:
        HTTPException: For invalid inputs or processing errors
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Expected image, got: {file.content_type}"
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            metrics.record_request(success=False, inference_time_ms=0.0)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run inference (synchronous CPU-bound operation)
        predictions, inference_time = model.predict(processed_image)
        
        # Record metrics
        metrics.record_request(success=True, inference_time_ms=inference_time)
        
        # Return response
        return InferenceResponse(
            predictions=predictions,
            inference_time_ms=round(inference_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        metrics.record_request(success=False, inference_time_ms=0.0)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during inference: {str(e)}"
        )
    finally:
        # Ensure file is closed
        await file.close()


@app.get(
    "/",
    response_model=ServiceInfo,
    tags=["info"],
    summary="Service Information",
    description="Get basic information about the service and available endpoints"
)
async def root() -> ServiceInfo:
    """Root endpoint with service information."""
    return ServiceInfo(
        service="Image Inference Service",
        version="1.0.0",
        endpoints={
            "health": "/health",
            "inference": "/infer (POST)",
            "batch_inference": "/infer/batch (POST)",
            "metrics": "/metrics",
            "reload_model": "/admin/reload (POST)",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    )


@app.post(
    "/infer/batch",
    response_model=BatchInferenceResponse,
    tags=["inference"],
    summary="Batch Image Classification",
    description="""
    Classify multiple images in a single request.
    
    Upload multiple image files and receive predictions for each.
    Failed images will be included in the response with error messages.
    
    Performance: Processing is done sequentially. For better throughput,
    consider parallel processing or GPU acceleration.
    """,
    responses={
        200: {
            "description": "Batch processing complete (may include partial failures)",
        },
        400: {
            "description": "No files provided",
        }
    }
)
async def batch_infer(
    files: List[UploadFile] = File(..., description="Multiple image files")
) -> BatchInferenceResponse:
    """
    Batch inference endpoint for processing multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        BatchInferenceResponse with results for each image
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append(BatchPrediction(
                    filename=file.filename or "unknown",
                    predictions=[],
                    inference_time_ms=0.0,
                    error=f"Invalid file type: {file.content_type}"
                ))
                failed += 1
                continue
            
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Validate image
            is_valid, error_msg = validate_image(image)
            if not is_valid:
                results.append(BatchPrediction(
                    filename=file.filename or "unknown",
                    predictions=[],
                    inference_time_ms=0.0,
                    error=error_msg
                ))
                failed += 1
                continue
            
            # Preprocess and infer
            processed_image = preprocess_image(image)
            predictions, inference_time = model.predict(processed_image)
            
            results.append(BatchPrediction(
                filename=file.filename or "unknown",
                predictions=predictions,
                inference_time_ms=round(inference_time, 2),
                error=None
            ))
            successful += 1
            metrics.record_request(success=True, inference_time_ms=inference_time)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append(BatchPrediction(
                filename=file.filename or "unknown",
                predictions=[],
                inference_time_ms=0.0,
                error=str(e)
            ))
            failed += 1
            metrics.record_request(success=False, inference_time_ms=0.0)
        finally:
            await file.close()
    
    total_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return BatchInferenceResponse(
        total_images=len(files),
        successful=successful,
        failed=failed,
        total_time_ms=round(total_time, 2),
        results=results
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["info"],
    summary="Service Metrics",
    description="Get performance metrics including request counts, latency statistics, and success rates"
)
async def get_metrics() -> MetricsResponse:
    """
    Get service performance metrics.
    
    Returns:
        MetricsResponse with comprehensive performance statistics
    """
    return MetricsResponse(**metrics.get_stats())


@app.post(
    "/admin/reload",
    tags=["admin"],
    summary="Reload Model",
    description="""
    Reload the ONNX model without restarting the service.
    
    Useful for:
    - Loading a new model version
    - Recovering from model corruption
    - Updating model configuration
    
    Note: This will briefly pause inference requests during reload.
    """,
    responses={
        200: {
            "description": "Model reloaded successfully",
            "content": {
                "application/json": {
                    "example": {"message": "Model reloaded successfully"}
                }
            }
        },
        500: {
            "description": "Failed to reload model"
        }
    }
)
async def reload_model() -> Dict[str, str]:
    """
    Reload the model without restarting the service.
    
    Returns:
        Success message
        
    Raises:
        HTTPException: If model reload fails
    """
    try:
        logger.info("Reloading model...")
        model.load()
        logger.info("Model reloaded successfully")
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )
