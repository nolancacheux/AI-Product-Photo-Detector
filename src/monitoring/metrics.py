"""Prometheus metrics for observability."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
APP_INFO = Info(
    "aidetect_app",
    "Application information",
)

# Model metrics
MODEL_INFO = Info(
    "aidetect_model",
    "Model information",
)
MODEL_LOAD_TIME = Gauge(
    "aidetect_model_load_seconds",
    "Time taken to load the model",
)
MODEL_LOADED = Gauge(
    "aidetect_model_loaded",
    "Whether the model is loaded (1) or not (0)",
)

# Prediction metrics
PREDICTIONS_TOTAL = Counter(
    "aidetect_predictions_total",
    "Total number of predictions",
    ["status", "prediction", "confidence"],
)
PREDICTION_LATENCY = Histogram(
    "aidetect_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_PROBABILITY = Histogram(
    "aidetect_prediction_probability",
    "Distribution of prediction probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Batch prediction metrics
BATCH_PREDICTIONS_TOTAL = Counter(
    "aidetect_batch_predictions_total",
    "Total number of batch prediction requests",
    ["status"],
)
BATCH_SIZE_HISTOGRAM = Histogram(
    "aidetect_batch_size",
    "Number of images in batch requests",
    buckets=[1, 2, 3, 5, 10, 15, 20],
)
BATCH_LATENCY = Histogram(
    "aidetect_batch_latency_seconds",
    "Batch prediction latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Image validation metrics
IMAGE_SIZE_BYTES = Histogram(
    "aidetect_image_size_bytes",
    "Size of uploaded images in bytes",
    buckets=[10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
)
IMAGE_DIMENSIONS = Histogram(
    "aidetect_image_dimension_pixels",
    "Image dimensions (max of width/height)",
    buckets=[100, 224, 512, 1024, 2048, 4096, 8192],
)
IMAGE_VALIDATION_ERRORS = Counter(
    "aidetect_image_validation_errors_total",
    "Total number of image validation errors",
    ["error_type"],
)

# Request/Response size metrics
REQUEST_SIZE_BYTES = Histogram(
    "aidetect_request_size_bytes",
    "Size of incoming HTTP request bodies in bytes",
    buckets=[1000, 10000, 100000, 500000, 1000000, 5000000, 10000000],
)
RESPONSE_SIZE_BYTES = Histogram(
    "aidetect_response_size_bytes",
    "Size of outgoing HTTP response bodies in bytes",
    buckets=[100, 500, 1000, 5000, 10000, 50000, 100000],
)

# Concurrent requests tracking
ACTIVE_REQUESTS = Gauge(
    "aidetect_active_requests",
    "Number of currently active requests",
)
CONCURRENT_REQUESTS_MAX = Gauge(
    "aidetect_concurrent_requests_max",
    "High watermark of concurrent requests since last reset",
)

# Rate limiting metrics
RATE_LIMIT_EXCEEDED = Counter(
    "aidetect_rate_limit_exceeded_total",
    "Total number of rate limit exceeded responses",
    ["endpoint"],
)

# Error metrics
ERRORS_TOTAL = Counter(
    "aidetect_errors_total",
    "Total number of errors",
    ["type", "endpoint"],
)

# HTTP metrics (per endpoint)
HTTP_REQUESTS_TOTAL = Counter(
    "aidetect_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION = Histogram(
    "aidetect_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Track current watermark internally
_current_active: int = 0
_max_active: int = 0


def track_request_start() -> None:
    """Track the start of a request, updating active and max gauges."""
    global _current_active, _max_active
    ACTIVE_REQUESTS.inc()
    _current_active += 1
    if _current_active > _max_active:
        _max_active = _current_active
        CONCURRENT_REQUESTS_MAX.set(_max_active)


def track_request_end() -> None:
    """Track the end of a request."""
    global _current_active
    ACTIVE_REQUESTS.dec()
    _current_active = max(0, _current_active - 1)


def set_app_info(version: str, environment: str = "production") -> None:
    """Set application info metrics.

    Args:
        version: Application version.
        environment: Deployment environment.
    """
    APP_INFO.info(
        {
            "version": version,
            "environment": environment,
        }
    )


def set_model_info(
    name: str,
    version: str,
    architecture: str,
    parameters: int,
) -> None:
    """Set model info metrics.

    Args:
        name: Model name.
        version: Model version.
        architecture: Model architecture.
        parameters: Number of parameters.
    """
    MODEL_INFO.info(
        {
            "name": name,
            "version": version,
            "architecture": architecture,
            "parameters": str(parameters),
        }
    )


def record_prediction(
    prediction: str,
    probability: float,
    confidence: str,
    latency_seconds: float,
    success: bool = True,
) -> None:
    """Record prediction metrics.

    Args:
        prediction: Prediction result.
        probability: Prediction probability.
        confidence: Confidence level.
        latency_seconds: Prediction latency.
        success: Whether prediction was successful.
    """
    status = "success" if success else "error"
    PREDICTIONS_TOTAL.labels(
        status=status,
        prediction=prediction,
        confidence=confidence,
    ).inc()

    if success:
        PREDICTION_LATENCY.observe(latency_seconds)
        PREDICTION_PROBABILITY.observe(probability)


def record_batch_prediction(
    batch_size: int,
    successful: int,
    failed: int,
    latency_seconds: float,
) -> None:
    """Record batch prediction metrics.

    Args:
        batch_size: Total number of images in batch.
        successful: Number of successful predictions.
        failed: Number of failed predictions.
        latency_seconds: Total batch latency.
    """
    BATCH_SIZE_HISTOGRAM.observe(batch_size)
    BATCH_LATENCY.observe(latency_seconds)
    status = "success" if failed == 0 else "partial"
    BATCH_PREDICTIONS_TOTAL.labels(status=status).inc()
