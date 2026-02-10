"""Route modules for the inference API."""

from src.inference.routes.health import router as health_router
from src.inference.routes.monitoring import router as monitoring_router
from src.inference.routes.predict import router as predict_router
from src.inference.routes.privacy import router as privacy_router

__all__ = [
    "health_router",
    "monitoring_router",
    "predict_router",
    "privacy_router",
]
