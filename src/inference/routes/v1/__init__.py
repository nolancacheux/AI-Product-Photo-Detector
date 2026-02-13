"""API v1 route modules.

Re-exports the core route modules for mounting under /v1 prefix.
Root-level routes are maintained separately for backward compatibility.
"""

from src.inference.routes.v1.info import router as info_router
from src.inference.routes.v1.monitoring import router as monitoring_router
from src.inference.routes.v1.predict import router as predict_router

__all__ = ["info_router", "monitoring_router", "predict_router"]
