"""Tests for caching module."""

import time

import pytest

from src.inference.cache import InMemoryCache, PredictionCache


class TestInMemoryCache:
    """Tests for in-memory cache."""

    @pytest.fixture
    def cache(self) -> InMemoryCache:
        """Create test cache."""
        return InMemoryCache(max_size=10, default_ttl=60)

    def test_set_and_get(self, cache: InMemoryCache) -> None:
        """Test basic set and get."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self, cache: InMemoryCache) -> None:
        """Test get nonexistent key returns None."""
        assert cache.get("nonexistent") is None

    def test_delete(self, cache: InMemoryCache) -> None:
        """Test delete removes key."""
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self, cache: InMemoryCache) -> None:
        """Test clear removes all keys."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        cache = InMemoryCache(max_size=10, default_ttl=1)
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when at capacity."""
        cache = InMemoryCache(max_size=3, default_ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key4") == "value4"

    def test_stats(self, cache: InMemoryCache) -> None:
        """Test stats tracking."""
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.stats()
        assert stats["type"] == "memory"
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestPredictionCache:
    """Tests for prediction cache."""

    @pytest.fixture
    def cache(self) -> PredictionCache:
        """Create test prediction cache."""
        return PredictionCache(backend="memory", ttl=60, max_memory_size=100)

    def test_generate_cache_key(self) -> None:
        """Test cache key generation."""
        key1 = PredictionCache.generate_cache_key(b"image1", "v1.0")
        key2 = PredictionCache.generate_cache_key(b"image1", "v1.0")
        key3 = PredictionCache.generate_cache_key(b"image1", "v2.0")
        key4 = PredictionCache.generate_cache_key(b"image2", "v1.0")

        # Same content + version = same key
        assert key1 == key2
        # Different version = different key
        assert key1 != key3
        # Different content = different key
        assert key1 != key4

    def test_set_and_get_prediction(self, cache: PredictionCache) -> None:
        """Test storing and retrieving predictions."""
        prediction = {
            "prediction": "ai_generated",
            "probability": 0.85,
            "confidence": "high",
        }
        cache.set_prediction(b"test_image", "v1.0", prediction)

        result = cache.get_prediction(b"test_image", "v1.0")
        assert result == prediction

    def test_cache_miss(self, cache: PredictionCache) -> None:
        """Test cache miss returns None."""
        result = cache.get_prediction(b"nonexistent", "v1.0")
        assert result is None

    def test_model_version_isolation(self, cache: PredictionCache) -> None:
        """Test different model versions have separate cache."""
        prediction_v1 = {"prediction": "real", "probability": 0.2}
        prediction_v2 = {"prediction": "ai_generated", "probability": 0.9}

        cache.set_prediction(b"test_image", "v1.0", prediction_v1)
        cache.set_prediction(b"test_image", "v2.0", prediction_v2)

        assert cache.get_prediction(b"test_image", "v1.0") == prediction_v1
        assert cache.get_prediction(b"test_image", "v2.0") == prediction_v2

    def test_stats(self, cache: PredictionCache) -> None:
        """Test stats method."""
        stats = cache.stats()
        assert "type" in stats
