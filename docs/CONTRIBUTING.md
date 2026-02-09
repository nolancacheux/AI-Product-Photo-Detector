# Contributing Guide

## Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/nolancacheux/AI-Product-Photo-Detector.git
cd mlops_project

# With uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,ui]"

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

### Formatting

We use **Ruff** for linting and formatting:

```bash
# Check linting
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

### Type Checking

We use **MyPy** for static type checking:

```bash
mypy src/
```

### Pre-commit Hooks

Pre-commit runs automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestAIImageDetector::test_model_creation
```

### Writing Tests

- Place tests in `tests/` directory
- Use `pytest` fixtures for shared setup
- Aim for >80% code coverage
- Test both success and error cases

## Git Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Examples**:
```
feat(api): add batch prediction endpoint
fix(model): handle empty image input
docs(readme): update installation instructions
test(api): add tests for health endpoint
```

### Pull Requests

1. Create a feature branch from `main`
2. Make your changes with atomic commits
3. Ensure tests pass: `pytest`
4. Ensure linting passes: `ruff check`
5. Update documentation if needed
6. Submit PR with clear description

## Project Structure

```
mlops_project/
├── src/
│   ├── training/     # Model training
│   ├── inference/    # API server
│   ├── ui/           # Streamlit app
│   └── utils/        # Shared utilities
├── tests/            # Unit tests
├── configs/          # Configuration files
├── docker/           # Dockerfiles
├── docs/             # Documentation
└── notebooks/        # Jupyter notebooks
```

## Documentation

- Update `README.md` for user-facing changes
- Update `docs/` for technical details
- Add docstrings to all public functions
- Use Google-style docstrings

## Release Process

1. Update version in `pyproject.toml` and `src/__init__.py`
2. Update `CHANGELOG.md`
3. Create release tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. GitHub Actions will build and publish Docker images
