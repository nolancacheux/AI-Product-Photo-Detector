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

# Explainability metrics
EXPLAIN_REQUESTS_TOTAL = Counter(
    "aidetect_explain_requests_total",
    "Total number of explanation requests",
    ["status"],
)
EXPLAIN_LATENCY = Histogram(
    "aidetect_explain_latency_seconds",
    "Explanation generation latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
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

# Cache metrics
CACHE_HITS = Counter(
    "aidetect_cache_hits_total",
    "Total number of cache hits",
)
CACHE_MISSES = Counter(
    "aidetect_cache_misses_total",
    "Total number of cache misses",
)
CACHE_SIZE = Gauge(
    "aidetect_cache_size",
    "Current number of items in cache",
)

# Rate limiting metrics
RATE_LIMIT_EXCEEDED = Counter(
    "aidetect_rate_limit_exceeded_total",
    "Total number of rate limit exceeded responses",
    ["endpoint"],
)

# Authentication metrics
AUTH_ATTEMPTS = Counter(
    "aidetect_auth_attempts_total",
    "Total number of authentication attempts",
    ["method", "result"],
)

# Error metrics
ERRORS_TOTAL = Counter(
    "aidetect_errors_total",
    "Total number of errors",
    ["type", "endpoint"],
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    "aidetect_active_requests",
    "Number of currently active requests",
)


def set_app_info(version: str, environment: str = "production") -> None:
    """Set application info metrics.

    Args:
        version: Application version.
        environment: Deployment environment.
    """
    APP_INFO.info({
        "version": version,
        "environment": environment,
    })


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
    MODEL_INFO.info({
        "name": name,
        "version": version,
        "architecture": architecture,
        "parameters": str(parameters),
    })


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
    latency_seconds: float,
    successful: int,
    failed: int,
) -> None:
    """Record batch prediction metrics.

    Args:
        batch_size: Number of images in batch.
        latency_seconds: Total batch latency.
        successful: Number of successful predictions.
        failed: Number of failed predictions.
    """
    status = "success" if failed == 0 else ("partial" if successful > 0 else "error")
    BATCH_PREDICTIONS_TOTAL.labels(status=status).inc()
    BATCH_SIZE_HISTOGRAM.observe(batch_size)
    BATCH_LATENCY.observe(latency_seconds)


def record_image_validation(
    size_bytes: int,
    width: int,
    height: int,
    error_type: str | None = None,
) -> None:
    """Record image validation metrics.

    Args:
        size_bytes: Image size in bytes.
        width: Image width.
        height: Image height.
        error_type: Validation error type if any.
    """
    IMAGE_SIZE_BYTES.observe(size_bytes)
    IMAGE_DIMENSIONS.observe(max(width, height))

    if error_type:
        IMAGE_VALIDATION_ERRORS.labels(error_type=error_type).inc()


def record_cache_access(hit: bool) -> None:
    """Record cache access metrics.

    Args:
        hit: Whether it was a cache hit.
    """
    if hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()


def record_auth_attempt(method: str, success: bool) -> None:
    """Record authentication attempt metrics.

    Args:
        method: Authentication method (api_key, jwt).
        success: Whether authentication was successful.
    """
    result = "success" if success else "failure"
    AUTH_ATTEMPTS.labels(method=method, result=result).inc()


def record_error(error_type: str, endpoint: str) -> None:
    """Record error metrics.

    Args:
        error_type: Type of error.
        endpoint: API endpoint where error occurred.
    """
    ERRORS_TOTAL.labels(type=error_type, endpoint=endpoint).inc()
