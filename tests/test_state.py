"""Tests for the shared application state module."""

import concurrent.futures
import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module-level counter before each test."""
    import src.inference.state as mod

    with mod._predictions_lock:
        mod._total_predictions = 0
    yield
    with mod._predictions_lock:
        mod._total_predictions = 0


class TestIncrementPredictions:
    """Tests for increment_predictions()."""

    def test_returns_one_on_first_call(self) -> None:
        from src.inference.state import increment_predictions

        assert increment_predictions() == 1

    def test_returns_incrementing_values(self) -> None:
        from src.inference.state import increment_predictions

        results = [increment_predictions() for _ in range(5)]
        assert results == [1, 2, 3, 4, 5]


class TestGetTotalPredictions:
    """Tests for get_total_predictions()."""

    def test_initially_zero(self) -> None:
        from src.inference.state import get_total_predictions

        assert get_total_predictions() == 0

    def test_reflects_increments(self) -> None:
        from src.inference.state import get_total_predictions, increment_predictions

        increment_predictions()
        increment_predictions()
        increment_predictions()
        assert get_total_predictions() == 3


class TestThreadSafety:
    """Verify counter correctness under concurrent access."""

    def test_concurrent_increments(self) -> None:
        from src.inference.state import get_total_predictions, increment_predictions

        num_threads = 10
        increments_per_thread = 100
        expected_total = num_threads * increments_per_thread

        barrier = threading.Barrier(num_threads)

        def worker() -> list[int]:
            barrier.wait()
            return [increment_predictions() for _ in range(increments_per_thread)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker) for _ in range(num_threads)]
            all_values: list[int] = []
            for future in concurrent.futures.as_completed(futures):
                all_values.extend(future.result())

        assert get_total_predictions() == expected_total
        assert len(set(all_values)) == expected_total  # all values unique
        assert sorted(all_values) == list(range(1, expected_total + 1))
