"""Tests for the routes package."""


class TestRoutesInit:
    """Tests for routes/__init__.py."""

    def test_routes_package_imports(self) -> None:
        """Importing the routes package should not raise."""
        import src.inference.routes  # noqa: F401

    def test_package_has_docstring(self) -> None:
        """The routes package should have a module docstring."""
        import src.inference.routes

        assert src.inference.routes.__doc__ is not None
