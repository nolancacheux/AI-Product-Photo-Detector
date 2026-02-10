"""API key authentication with support for REQUIRE_AUTH enforcement."""

import hashlib
import hmac
import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from src.utils.logger import get_logger

logger = get_logger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key(prefix: str = "apd") -> str:
    """Generate a secure random API key.

    Utility for CLI / admin scripts. Not called by the API itself.

    Args:
        prefix: Short prefix for key identification.

    Returns:
        API key string like 'apd_a1b2c3d4e5f6...'
    """
    return f"{prefix}_{secrets.token_urlsafe(32)}"


class APIKeyManager:
    """API key manager with secure comparison and REQUIRE_AUTH support."""

    def __init__(self) -> None:
        """Initialize API key manager."""
        self._keys: set[str] = set()
        self._require_auth = os.getenv("REQUIRE_AUTH", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from environment."""
        keys_str = os.getenv("API_KEYS", "")
        if keys_str:
            for key in keys_str.split(","):
                key = key.strip()
                if key:
                    self._keys.add(self._hash_key(key))
            logger.info("API keys loaded", count=len(self._keys))

        if self._require_auth and not self._keys:
            logger.warning(
                "REQUIRE_AUTH is enabled but no API_KEYS configured — "
                "ALL requests will be rejected until API_KEYS is set"
            )

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_key(self, key: str) -> bool:
        """Validate an API key using constant-time comparison.

        Iterates ALL stored keys to avoid timing side-channels that
        leak the number of configured keys.
        """
        key_hash = self._hash_key(key)
        found = False
        for stored in self._keys:
            if hmac.compare_digest(key_hash, stored):
                found = True
        return found

    @property
    def is_enabled(self) -> bool:
        """Check if authentication is enabled (keys present OR REQUIRE_AUTH)."""
        return len(self._keys) > 0 or self._require_auth


# Global instance
api_key_manager = APIKeyManager()


async def verify_api_key(
    api_key: Annotated[str | None, Depends(API_KEY_HEADER)] = None,
) -> bool:
    """Verify API key if authentication is enabled.

    Returns True if:
    - Auth is disabled (no API_KEYS and REQUIRE_AUTH=false), OR
    - Valid API key provided

    Raises HTTPException if auth enabled but key invalid/missing.
    """
    if not api_key_manager.is_enabled:
        return True

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Set X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not api_key_manager.validate_key(api_key):
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True
