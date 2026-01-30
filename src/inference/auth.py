"""Authentication middleware for API security."""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_SCHEME = HTTPBearer(auto_error=False)
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class TokenData(BaseModel):
    """JWT token payload data."""

    sub: str
    exp: datetime
    iat: datetime
    scopes: list[str] = []


class APIKeyManager:
    """Manages API keys for authentication."""

    def __init__(self) -> None:
        """Initialize API key manager."""
        self._keys: dict[str, dict] = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from environment."""
        # Load keys from environment variable (comma-separated)
        keys_str = os.getenv("API_KEYS", "")
        if keys_str:
            for key in keys_str.split(","):
                key = key.strip()
                if key:
                    key_hash = self._hash_key(key)
                    self._keys[key_hash] = {
                        "name": f"key_{len(self._keys)}",
                        "scopes": ["predict", "batch", "explain"],
                        "created": datetime.now(timezone.utc),
                    }
            logger.info(f"Loaded {len(self._keys)} API keys from environment")

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key for secure storage.

        Args:
            key: Raw API key.

        Returns:
            Hashed key.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_key(self, key: str) -> dict | None:
        """Validate an API key.

        Args:
            key: Raw API key to validate.

        Returns:
            Key metadata if valid, None otherwise.
        """
        key_hash = self._hash_key(key)
        return self._keys.get(key_hash)

    def generate_key(self, name: str, scopes: list[str] | None = None) -> str:
        """Generate a new API key.

        Args:
            name: Key identifier name.
            scopes: Allowed scopes for this key.

        Returns:
            Generated API key.
        """
        key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(key)
        self._keys[key_hash] = {
            "name": name,
            "scopes": scopes or ["predict"],
            "created": datetime.now(timezone.utc),
        }
        return key

    @property
    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return len(self._keys) > 0 or os.getenv("REQUIRE_AUTH", "false").lower() == "true"


# Global API key manager
api_key_manager = APIKeyManager()


def create_jwt_token(subject: str, scopes: list[str] | None = None) -> str:
    """Create a JWT token.

    Args:
        subject: Token subject (user/client identifier).
        scopes: Allowed scopes.

    Returns:
        Encoded JWT token.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": now,
        "scopes": scopes or ["predict"],
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> TokenData | None:
    """Decode and validate a JWT token.

    Args:
        token: Encoded JWT token.

    Returns:
        Token data if valid, None otherwise.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(
            sub=payload["sub"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            scopes=payload.get("scopes", []),
        )
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None


async def get_api_key(
    api_key: Annotated[str | None, Depends(API_KEY_HEADER)] = None,
) -> str | None:
    """Extract API key from header.

    Args:
        api_key: API key from header.

    Returns:
        API key if present.
    """
    return api_key


async def get_bearer_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(BEARER_SCHEME)] = None,
) -> str | None:
    """Extract bearer token from header.

    Args:
        credentials: Bearer credentials.

    Returns:
        Token if present.
    """
    if credentials:
        return credentials.credentials
    return None


async def verify_auth(
    api_key: Annotated[str | None, Depends(get_api_key)] = None,
    bearer_token: Annotated[str | None, Depends(get_bearer_token)] = None,
    required_scope: str = "predict",
) -> dict:
    """Verify authentication via API key or JWT.

    Args:
        api_key: API key from header.
        bearer_token: JWT bearer token.
        required_scope: Required scope for this operation.

    Returns:
        Authentication context with user info and scopes.

    Raises:
        HTTPException: If authentication fails.
    """
    # If auth not enabled, allow all
    if not api_key_manager.is_enabled:
        return {"authenticated": False, "scopes": ["*"]}

    # Try API key first
    if api_key:
        key_data = api_key_manager.validate_key(api_key)
        if key_data:
            if required_scope not in key_data["scopes"] and "*" not in key_data["scopes"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required scope: {required_scope}",
                )
            return {
                "authenticated": True,
                "method": "api_key",
                "name": key_data["name"],
                "scopes": key_data["scopes"],
            }

    # Try JWT token
    if bearer_token:
        token_data = decode_jwt_token(bearer_token)
        if token_data:
            if required_scope not in token_data.scopes and "*" not in token_data.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required scope: {required_scope}",
                )
            return {
                "authenticated": True,
                "method": "jwt",
                "subject": token_data.sub,
                "scopes": token_data.scopes,
            }

    # No valid auth provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication. Provide X-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_scope(scope: str):
    """Create a dependency that requires a specific scope.

    Args:
        scope: Required scope.

    Returns:
        Dependency function.
    """

    async def _verify(
        api_key: Annotated[str | None, Depends(get_api_key)] = None,
        bearer_token: Annotated[str | None, Depends(get_bearer_token)] = None,
    ) -> dict:
        return await verify_auth(api_key, bearer_token, required_scope=scope)

    return _verify
