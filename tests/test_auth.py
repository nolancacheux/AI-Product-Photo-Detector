"""Tests for API key authentication module."""

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.inference.auth import APIKeyManager, verify_api_key


class TestAPIKeyManager:
    """Tests for APIKeyManager class."""

    def test_init_no_keys(self) -> None:
        """Manager with no API_KEYS env should have no keys."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("API_KEYS", None)
            manager = APIKeyManager()
            assert not manager.is_enabled
            assert len(manager._keys) == 0

    def test_init_with_single_key(self) -> None:
        """Manager should load a single API key."""
        with patch.dict(os.environ, {"API_KEYS": "test-key-123"}):
            manager = APIKeyManager()
            assert manager.is_enabled
            assert len(manager._keys) == 1

    def test_init_with_multiple_keys(self) -> None:
        """Manager should load comma-separated API keys."""
        with patch.dict(os.environ, {"API_KEYS": "key1,key2,key3"}):
            manager = APIKeyManager()
            assert manager.is_enabled
            assert len(manager._keys) == 3

    def test_init_with_whitespace_keys(self) -> None:
        """Manager should strip whitespace from keys."""
        with patch.dict(os.environ, {"API_KEYS": " key1 , key2 , key3 "}):
            manager = APIKeyManager()
            assert len(manager._keys) == 3

    def test_init_ignores_empty_keys(self) -> None:
        """Manager should ignore empty key segments."""
        with patch.dict(os.environ, {"API_KEYS": "key1,,key2,,,key3"}):
            manager = APIKeyManager()
            assert len(manager._keys) == 3

    def test_validate_correct_key(self) -> None:
        """Correct key should validate."""
        with patch.dict(os.environ, {"API_KEYS": "my-secret-key"}):
            manager = APIKeyManager()
            assert manager.validate_key("my-secret-key") is True

    def test_validate_wrong_key(self) -> None:
        """Wrong key should not validate."""
        with patch.dict(os.environ, {"API_KEYS": "my-secret-key"}):
            manager = APIKeyManager()
            assert manager.validate_key("wrong-key") is False

    def test_validate_empty_key(self) -> None:
        """Empty key should not validate."""
        with patch.dict(os.environ, {"API_KEYS": "my-secret-key"}):
            manager = APIKeyManager()
            assert manager.validate_key("") is False

    def test_is_enabled_with_keys(self) -> None:
        """is_enabled should be True when keys are configured."""
        with patch.dict(os.environ, {"API_KEYS": "key1"}):
            manager = APIKeyManager()
            assert manager.is_enabled is True

    def test_is_enabled_without_keys(self) -> None:
        """is_enabled should be False when no keys are configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("API_KEYS", None)
            manager = APIKeyManager()
            assert manager.is_enabled is False

    def test_hash_key_deterministic(self) -> None:
        """Same key should produce same hash."""
        hash1 = APIKeyManager._hash_key("test-key")
        hash2 = APIKeyManager._hash_key("test-key")
        assert hash1 == hash2

    def test_hash_key_different_keys(self) -> None:
        """Different keys should produce different hashes."""
        hash1 = APIKeyManager._hash_key("key-a")
        hash2 = APIKeyManager._hash_key("key-b")
        assert hash1 != hash2


class TestVerifyApiKey:
    """Tests for verify_api_key dependency."""

    @pytest.mark.asyncio
    async def test_no_auth_required(self) -> None:
        """When auth disabled, should return True."""
        with patch("src.inference.auth.api_key_manager") as mock_manager:
            mock_manager.is_enabled = False
            result = await verify_api_key(api_key=None)
            assert result is True

    @pytest.mark.asyncio
    async def test_no_auth_with_key_provided(self) -> None:
        """When auth disabled, should return True even with key."""
        with patch("src.inference.auth.api_key_manager") as mock_manager:
            mock_manager.is_enabled = False
            result = await verify_api_key(api_key="some-key")
            assert result is True

    @pytest.mark.asyncio
    async def test_auth_required_no_key(self) -> None:
        """When auth enabled but no key, should raise 401."""
        with patch("src.inference.auth.api_key_manager") as mock_manager:
            mock_manager.is_enabled = True
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key=None)
            assert exc_info.value.status_code == 401
            assert "API key required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_auth_required_invalid_key(self) -> None:
        """When auth enabled with invalid key, should raise 401."""
        with patch("src.inference.auth.api_key_manager") as mock_manager:
            mock_manager.is_enabled = True
            mock_manager.validate_key.return_value = False
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="bad-key")
            assert exc_info.value.status_code == 401
            assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_auth_required_valid_key(self) -> None:
        """When auth enabled with valid key, should return True."""
        with patch("src.inference.auth.api_key_manager") as mock_manager:
            mock_manager.is_enabled = True
            mock_manager.validate_key.return_value = True
            result = await verify_api_key(api_key="good-key")
            assert result is True
