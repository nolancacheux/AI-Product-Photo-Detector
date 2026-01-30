"""Response caching for prediction results."""

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def stats(self) -> dict:
        """Get cache statistics."""
        pass


class InMemoryCache(CacheBackend):
    """Simple in-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600) -> None:
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of items in cache.
            default_ttl: Default TTL in seconds.
        """
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def _is_expired(self, expiry: float) -> bool:
        """Check if a cache entry is expired."""
        return expiry > 0 and time.time() > expiry

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, (_, expiry) in self._cache.items() if expiry > 0 and now > expiry]
        for key in expired:
            del self._cache[key]

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        if key not in self._cache:
            self._misses += 1
            return None

        value, expiry = self._cache[key]

        if self._is_expired(expiry):
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds.
        """
        if ttl is None:
            ttl = self._default_ttl

        expiry = time.time() + ttl if ttl > 0 else 0

        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        self._evict_expired()
        total = self._hits + self._misses
        return {
            "type": "memory",
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


class RedisCache(CacheBackend):
    """Redis-backed cache."""

    def __init__(
        self,
        url: str | None = None,
        prefix: str = "aidetect:",
        default_ttl: int = 3600,
    ) -> None:
        """Initialize Redis cache.

        Args:
            url: Redis URL (defaults to REDIS_URL env var).
            prefix: Key prefix for namespacing.
            default_ttl: Default TTL in seconds.
        """
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._client = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis

            self._client = redis.from_url(self._url)
            self._client.ping()
            logger.info("Connected to Redis", url=self._url)
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self._client = None

    def _key(self, key: str) -> str:
        """Generate prefixed key."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        if not self._client:
            return None

        try:
            value = self._client.get(self._key(key))
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in Redis."""
        if not self._client:
            return

        if ttl is None:
            ttl = self._default_ttl

        try:
            self._client.setex(
                self._key(key),
                ttl,
                json.dumps(value),
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> None:
        """Delete value from Redis."""
        if not self._client:
            return

        try:
            self._client.delete(self._key(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    def clear(self) -> None:
        """Clear all cached values with prefix."""
        if not self._client:
            return

        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(
                    cursor=cursor,
                    match=f"{self._prefix}*",
                    count=100,
                )
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def stats(self) -> dict:
        """Get cache statistics."""
        if not self._client:
            return {"type": "redis", "connected": False}

        try:
            info = self._client.info("stats")
            return {
                "type": "redis",
                "connected": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys": self._client.dbsize(),
            }
        except Exception as e:
            return {"type": "redis", "connected": False, "error": str(e)}


class PredictionCache:
    """High-level cache manager for predictions."""

    def __init__(
        self,
        backend: str = "auto",
        ttl: int = 3600,
        max_memory_size: int = 1000,
    ) -> None:
        """Initialize prediction cache.

        Args:
            backend: Backend type ('memory', 'redis', 'auto').
            ttl: Default TTL in seconds.
            max_memory_size: Max items for memory backend.
        """
        self._ttl = ttl

        if backend == "auto":
            # Try Redis first, fall back to memory
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                self._backend = RedisCache(url=redis_url, default_ttl=ttl)
                if self._backend._client:
                    logger.info("Using Redis cache backend")
                else:
                    self._backend = InMemoryCache(max_size=max_memory_size, default_ttl=ttl)
                    logger.info("Falling back to memory cache")
            else:
                self._backend = InMemoryCache(max_size=max_memory_size, default_ttl=ttl)
                logger.info("Using in-memory cache backend")
        elif backend == "redis":
            self._backend = RedisCache(default_ttl=ttl)
        else:
            self._backend = InMemoryCache(max_size=max_memory_size, default_ttl=ttl)

    @staticmethod
    def generate_cache_key(image_bytes: bytes, model_version: str) -> str:
        """Generate cache key from image content and model version.

        Args:
            image_bytes: Raw image bytes.
            model_version: Model version string.

        Returns:
            Cache key string.
        """
        content_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        return f"pred:{model_version}:{content_hash}"

    def get_prediction(self, image_bytes: bytes, model_version: str) -> dict | None:
        """Get cached prediction for image.

        Args:
            image_bytes: Raw image bytes.
            model_version: Model version string.

        Returns:
            Cached prediction dict or None.
        """
        key = self.generate_cache_key(image_bytes, model_version)
        result = self._backend.get(key)
        if result:
            logger.debug("Cache hit", key=key)
        return result

    def set_prediction(
        self,
        image_bytes: bytes,
        model_version: str,
        prediction: dict,
        ttl: int | None = None,
    ) -> None:
        """Cache prediction result.

        Args:
            image_bytes: Raw image bytes.
            model_version: Model version string.
            prediction: Prediction result dict.
            ttl: Optional custom TTL.
        """
        key = self.generate_cache_key(image_bytes, model_version)
        self._backend.set(key, prediction, ttl or self._ttl)
        logger.debug("Cache set", key=key)

    def stats(self) -> dict:
        """Get cache statistics."""
        return self._backend.stats()

    def clear(self) -> None:
        """Clear all cached predictions."""
        self._backend.clear()


# Global cache instance
prediction_cache: PredictionCache | None = None


def get_prediction_cache() -> PredictionCache:
    """Get or create the global prediction cache."""
    global prediction_cache
    if prediction_cache is None:
        prediction_cache = PredictionCache()
    return prediction_cache
