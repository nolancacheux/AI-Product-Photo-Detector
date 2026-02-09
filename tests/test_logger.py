"""Tests for the logger module."""

from src.utils.logger import (
    clear_request_id,
    get_logger,
    get_request_id,
    set_request_id,
    setup_logging,
)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_bound_logger(self) -> None:
        """get_logger should return a structlog logger proxy."""
        logger = get_logger("test")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_different_names_return_loggers(self) -> None:
        """Different names should still return valid loggers."""
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not None
        assert logger2 is not None


class TestRequestId:
    """Tests for request ID context management."""

    def test_set_and_get_request_id(self) -> None:
        """Setting a request ID should be retrievable."""
        set_request_id("abc123")
        assert get_request_id() == "abc123"
        clear_request_id()

    def test_set_generates_id_when_none(self) -> None:
        """set_request_id(None) should auto-generate an ID."""
        generated = set_request_id(None)
        assert generated is not None
        assert len(generated) == 8
        assert get_request_id() == generated
        clear_request_id()

    def test_clear_request_id(self) -> None:
        """clear_request_id should reset to None."""
        set_request_id("temp")
        clear_request_id()
        assert get_request_id() is None

    def test_default_request_id_is_none(self) -> None:
        """Default request ID should be None."""
        clear_request_id()
        assert get_request_id() is None


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_json_format(self) -> None:
        """setup_logging with json_format=True should not crash."""
        setup_logging(level="DEBUG", json_format=True, include_service_info=True)

    def test_console_format(self) -> None:
        """setup_logging with json_format=False should not crash."""
        setup_logging(level="INFO", json_format=False, include_service_info=False)

    def test_warning_level(self) -> None:
        """setup_logging with WARNING level should not crash."""
        setup_logging(level="WARNING", json_format=True)

    def test_without_service_info(self) -> None:
        """setup_logging without service info should not crash."""
        setup_logging(level="INFO", json_format=True, include_service_info=False)
