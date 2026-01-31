"""Simple API key authentication."""

import hashlib
import os
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from src.utils.logger import get_logger

logger = get_logger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyManager:
    """Simple API key manager."""

    def __init__(self) -> None:
        """Initialize API key manager."""
        self._keys: set[str] = set()
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from environment."""
        keys_str = os.getenv("API_KEYS", "")
        if keys_str:
            for key in keys_str.split(","):
                key = key.strip()
                if key:
                    self._keys.add(self._hash_key(key))
            logger.info(f"Loaded {len(self._keys)} API keys")

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key for secure comparison."""
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_key(self, key: str) -> bool:
        """Validate an API key."""
        return self._hash_key(key) in self._keys

    @property
    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return len(self._keys) > 0


# Global instance
api_key_manager = APIKeyManager()


async def verify_api_key(
    api_key: Annotated[str | None, Depends(API_KEY_HEADER)] = None,
) -> bool:
    """Verify API key if authentication is enabled.

    Returns True if:
    - Auth is disabled (no API_KEYS configured), OR
    - Valid API key provided

    Raises HTTPException if auth enabled but key invalid/missing.
    """
    if not api_key_manager.is_enabled:
        return True

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Set X-API-Key header.",
        )

    if not api_key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return True
