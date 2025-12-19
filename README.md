# Image Inference Service

A production-ready FastAPI service for image classification using ResNet-50 ONNX model. Includes batch inference, metrics tracking, hot model reload, and Docker deployment.

## Features

- **Fast Inference**: ~15-50ms per image on CPU
- **Batch Processing**: Process multiple images in a single request
- **Performance Metrics**: Track requests, latency, and success rates
- **Hot Model Reload**: Update model without service restart
- **Docker Ready**: Multi-stage builds and docker-compose setup
- **Interactive API Docs**: Swagger UI and ReDoc
- **Production Ready**: Health checks, logging, error handling

## Quick Start

### Prerequisites

- Python 3.12+
- pip (or Docker for containerized deployment)

### Local Development Setup

1. **Clone and navigate to the project:**
```bash
cd Youverse
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create environment file:**
```bash
cp .env.example .env
```

5. **Run the service:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

### Docker Deployment

All Docker-related files are organized in the `docker/` directory for better project structure.

1. **Build and run with Docker Compose:**
```bash
cd docker
docker-compose up -d
```

2. **Check service status:**
```bash
cd docker
docker-compose ps
docker-compose logs -f
```

3. **Stop the service:**
```bash
cd docker
docker-compose down
```

Alternatively, use Docker directly:
```bash
docker build -f docker/Dockerfile -t inference-api .
docker run -p 8000:8000 inference-api
```

## API Usage

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Endpoints

#### 1. Single Image Classification

**POST /infer**

Classify a single image and get top-k predictions.

```bash
curl -X POST "http://localhost:8000/infer" \
  -F "file=@animals/cat.jpg"
```

**Response:**
```json
{
  "predictions": [
    {"label": "tabby_cat", "confidence": 0.87},
    {"label": "tiger_cat", "confidence": 0.08},
    {"label": "Egyptian_cat", "confidence": 0.03}
  ],
  "inference_time_ms": 12.5
}
```

#### 2. Batch Image Classification

**POST /infer/batch**

Process multiple images in a single request.

```bash
curl -X POST "http://localhost:8000/infer/batch" \
  -F "files=@animals/cat.jpg" \
  -F "files=@animals/dog.jpg" \
  -F "files=@animals/bird.jpg"
```

**Response:**
```json
{
  "total_images": 3,
  "successful": 3,
  "failed": 0,
  "total_time_ms": 45.3,
  "results": [
    {
      "filename": "cat.jpg",
      "predictions": [{"label": "tabby_cat", "confidence": 0.87}],
      "inference_time_ms": 12.5,
      "error": null
    },
    ...
  ]
}
```

#### 3. Health Check

**GET /health**

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### 4. Performance Metrics

**GET /metrics**

Get service performance statistics.

```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
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
```

#### 5. Model Reload

**POST /admin/reload**

Reload the model without restarting the service.

```bash
curl -X POST http://localhost:8000/admin/reload
```

**Response:**
```json
{
  "message": "Model reloaded successfully"
}
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
```

## Configuration

Environment variables (`.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to ONNX model file | `models/resnet50.onnx` |
| `LABELS_PATH` | Path to ImageNet labels | `models/imagenet_classes.txt` |
| `TOP_K` | Number of predictions to return | `3` |
| `NUM_THREADS` | ONNX Runtime thread count | `4` |


## Architecture & Design Decisions

### Project Structure
```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app & endpoints
│   ├── inference.py         # ONNX model wrapper
│   ├── preprocessing.py     # Image preprocessing
│   ├── schemas.py           # Pydantic models
│   ├── batch_schemas.py     # Batch & metrics schemas
│   ├── metrics.py           # Performance tracking
│   └── config.py            # Configuration
├── tests/
│   ├── test_api.py          # API integration tests
│   ├── test_inference.py    # Inference logic tests
│   └── test_preprocessing.py
├── models/
│   ├── resnet50.onnx        # ONNX model (97.6 MB)
│   └── imagenet_classes.txt
├── docker/
│   ├── Dockerfile           # Multi-stage Docker build
│   └── docker-compose.yml   # Container orchestration
├── requirements.txt
├── .env.example
└── README.md
```

### Key Design Decisions

#### 1. Model Loading at Startup
- **Decision**: Load ONNX model once during application startup
- **Rationale**: Avoids expensive loading overhead on each request
- **Implementation**: FastAPI lifespan events (modern approach)
- **Trade-off**: Slower startup (~500ms), but 10-30ms inference vs ~500ms+ if loading per request

#### 2. Async vs Sync Hybrid Approach
- **Decision**: Async endpoints with synchronous inference
- **Rationale**: 
  - ONNX Runtime CPU provider is CPU-bound (no async benefit)
  - Async wrapper prevents blocking during I/O (file upload/download)
  - FastAPI handles multiple concurrent requests efficiently
- **Trade-off**: Can't parallelize inference itself, but maintains non-blocking I/O

#### 3. Separation of Concerns
- **Preprocessing** (`preprocessing.py`): Image validation and transformation
- **Inference** (`inference.py`): Model loading and prediction logic
- **API Layer** (`main.py`): HTTP interface and routing
- **Metrics** (`metrics.py`): Thread-safe performance tracking
- **Schemas** (`schemas.py`, `batch_schemas.py`): Type validation and API docs
- **Config** (`config.py`): Centralized environment-based configuration
- **Rationale**: Single responsibility principle, testability, maintainability

#### 4. Batch Inference Implementation
- **Decision**: Sequential processing in current implementation
- **Why Sequential**: 
  - Simpler code, easier to reason about
  - ONNX CPU provider doesn't benefit from threading for small batches
  - Individual error handling for each image
- **Future Optimization**: Could add true batch processing with GPU provider or parallel CPU threads

#### 5. Metrics Design
- **Thread-safe**: Uses locks to prevent race conditions
- **Memory-efficient**: Keeps only last 1000 latencies
- **Comprehensive**: Tracks count, success rate, latency percentiles (p95, p99)
- **Trade-off**: Slight overhead on each request (~microseconds) for valuable observability

#### 6. Hot Model Reload
- **Decision**: Provide admin endpoint to reload model without restart
- **Use Cases**:
  - Model version updates
  - A/B testing
  - Recovery from corruption
- **Trade-off**: Brief pause during reload, but no downtime

#### 7. Docker Multi-stage Build
- **Decision**: Builder stage + slim runtime stage
- **Benefits**:
  - Smaller final image (~500MB vs ~1.5GB)
  - No build tools in production image
  - Better security (fewer attack surfaces)
- **Trade-off**: Slightly longer build time

#### 8. Error Handling Strategy
- Input validation at multiple levels:
  - File type validation (MIME type)
  - Image format validation (PIL can open)
  - Dimension validation (reasonable sizes)
- Proper HTTP status codes:
  - 400 for client errors (bad input)
  - 500 for server errors (internal issues)
- Informative error messages without exposing internals
- Comprehensive logging for debugging

### Image Preprocessing Pipeline

ResNet-50 expects specific input format:

1. **Resize**: 224×224 pixels (BILINEAR interpolation)
2. **Color Space**: RGB (auto-converts grayscale/RGBA)
3. **Normalization**: ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
4. **Format**: NCHW (Batch, Channels, Height, Width)
5. **Data Type**: Float32

**Design Choice**: Preprocessing happens per-request rather than caching because:
- Images are typically unique
- Preprocessing is fast (~5-10ms)
- Caching adds complexity and memory overhead

### Performance Characteristics

On modern CPU (Apple M1/M2 or Intel i7):
- **Model Loading**: ~500ms (one-time at startup)
- **Image Preprocessing**: 5-10ms
- **Inference (CPU)**: 10-30ms
- **Total Request Latency**: 15-40ms
- **Throughput**: ~25-60 requests/second (single worker)

**Bottleneck**: Inference is CPU-bound. Scaling options:
1. Horizontal: Multiple Uvicorn workers
2. Vertical: GPU acceleration (CUDA provider)
3. Batch: True batch inference for higher throughput

## What Would I Improve With More Time

### 1. GPU Acceleration (High Priority)
**Current**: CPU-only inference (~20-30ms)
**With GPU**: Could achieve 2-5ms inference time

```python
# Would add GPU support in inference.py
import onnxruntime as ort

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
```

**Benefits**: 5-10x faster inference, better throughput
**Trade-off**: Requires NVIDIA GPU, larger Docker image

### 2. True Batch Inference Optimization
**Current**: Sequential processing in batch endpoint
**Improved**: Actual batched ONNX inference

```python
# Process all images as single batch tensor
batch_tensor = np.stack([preprocess_image(img) for img in images])
predictions = model.predict_batch(batch_tensor)  # Single inference call
```

**Benefits**: 2-3x throughput for batch requests
**Use Case**: Bulk image processing, video frame analysis

### 3. Intelligent Caching
**Implementation**: Redis-based caching with image hashing

```python
from hashlib import sha256
image_hash = sha256(image_bytes).hexdigest()
cached_result = redis.get(f"inference:{image_hash}")
```

**Benefits**: Instant results for duplicate images
**Use Case**: Processing same images multiple times (thumbnails, previews)

### 4. Advanced Monitoring
**Would Add**:
- Prometheus metrics exporter
- OpenTelemetry distributed tracing
- Grafana dashboards
- Alert rules (high latency, error rate spikes)

```python
from prometheus_client import Counter, Histogram
inference_duration = Histogram('inference_duration_seconds', 'Inference time')
```

### 5. Authentication & Rate Limiting
**Security Improvements**:
- API key authentication
- Per-user rate limits
- Usage quotas and billing
- JWT tokens for service-to-service auth

```python
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")
```

### 6. Model Versioning System
**Features**:
- Multiple model versions running simultaneously
- A/B testing framework
- Gradual rollout (canary deployments)
- Automatic rollback on performance degradation

```python
@app.post("/infer")
async def infer(file: UploadFile, model_version: str = "latest"):
    model = get_model(model_version)  # v1, v2, latest, etc.
```

### 7. Advanced Image Preprocessing
**Enhancements**:
- EXIF-based auto-rotation
- Smart cropping (face detection, saliency maps)
- Format-specific optimizations (WebP, HEIC support)
- Adaptive quality/size based on network conditions

### 8. Load Testing & Performance Tuning
**Would Implement**:
- Locust/k6 load tests
- Profile with py-spy/cProfile
- Optimize hot paths
- Memory profiling (tracemalloc)
- Determine optimal worker count

### 9. Production Deployment Enhancements
**Infrastructure**:
- Kubernetes manifests (Deployment, Service, HPA)
- Horizontal Pod Autoscaling based on CPU/custom metrics
- Readiness/liveness probes
- Blue-green deployment strategy
- Multi-region deployment

**Example HPA**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    name: inference-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: inference_latency_p99
      target:
        value: "50"  # Scale up if p99 > 50ms
```

### 10. Enhanced Error Recovery
**Improvements**:
- Circuit breaker pattern (fail fast when model unavailable)
- Exponential backoff for retries
- Graceful degradation (fallback to simpler model)
- Dead letter queue for failed requests

## Troubleshooting

### Model file not found
```bash
ls -lh models/
# Should show resnet50.onnx (97.6 MB) and imagenet_classes.txt
```

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

### Port already in use
```bash
# Change port
uvicorn app.main:app --port 8001

# Or find and kill process
lsof -ti:8000 | xargs kill -9
```

### Docker build fails
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
cd docker
docker-compose build --no-cache
```

### Out of memory
```bash
# Reduce thread count in .env
NUM_THREADS=2

# Or limit Docker memory
docker run -m 2g -p 8000:8000 inference-api
```

## Acknowledgments

- ResNet-50 model from [ONNX Model Zoo](https://github.com/onnx/models)
- ImageNet labels from [PyTorch Hub](https://github.com/pytorch/hub)
- FastAPI framework by Sebastián Ramírez
