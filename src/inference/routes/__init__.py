"""Route modules for the inference API."""

from src.inference.routes.info import router as info_router
from src.inference.routes.monitoring import router as monitoring_router
from src.inference.routes.predict import router as predict_router

__all__ = ["info_router", "monitoring_router", "predict_router"]
