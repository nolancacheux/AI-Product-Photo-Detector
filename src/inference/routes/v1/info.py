"""V1 informational routes — thin wrapper over shared implementation."""

from typing import Any

from fastapi import APIRouter

from src.inference.routes.info import privacy as _privacy
from src.inference.routes.info import root as _root

router = APIRouter(tags=["Info"])


@router.get("/privacy")
async def privacy_v1() -> dict[str, Any]:
    """Privacy policy summary (v1)."""
    return await _privacy()


@router.get("/")
async def root_v1() -> dict[str, Any]:
    """API root endpoint (v1)."""
    return await _root()
