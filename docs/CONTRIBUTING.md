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
cd AI-Product-Photo-Detector

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

We use **mypy** for static type checking:

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

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
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
AI-Product-Photo-Detector/
├── .github/workflows/      # CI/CD pipelines
├── src/
│   ├── data/               # Data download and validation
│   ├── inference/          # API server
│   │   ├── routes/         # API route handlers
│   │   ├── api.py          # FastAPI application
│   │   ├── predictor.py    # Model loading and inference
│   │   ├── explainer.py    # Grad-CAM heatmap generation
│   │   ├── auth.py         # API key authentication
│   │   ├── validation.py   # Input validation
│   │   ├── schemas.py      # Pydantic models
│   │   ├── shadow.py       # Shadow model comparison
│   │   ├── state.py        # Application state
│   │   └── rate_limit.py   # Rate limiting
│   ├── training/           # Model training
│   │   ├── train.py        # Training loop with MLflow
│   │   ├── model.py        # EfficientNet-B0 architecture
│   │   ├── dataset.py      # PyTorch dataset
│   │   ├── augmentation.py # Data augmentation
│   │   ├── gcs.py          # GCS integration
│   │   └── vertex_submit.py # Vertex AI job submission
│   ├── pipelines/          # Pipeline orchestration
│   ├── monitoring/         # Observability
│   ├── ui/                 # Streamlit web interface
│   └── utils/              # Shared utilities
├── tests/                  # Unit and integration tests
├── configs/                # Configuration files
├── docker/                 # Dockerfiles
├── terraform/              # Infrastructure as Code
├── scripts/                # Data download utilities
├── notebooks/              # Jupyter notebooks (Colab training)
├── dvc.yaml                # DVC pipeline definition
├── docker-compose.yml      # Local development stack
├── Makefile                # Development commands
└── pyproject.toml          # Python dependencies
```

## Makefile Commands

```bash
make help          # List all commands
make dev           # Install dev dependencies + pre-commit
make lint          # Ruff + mypy
make format        # Auto-format code
make test          # Run pytest with coverage
make data          # Download CIFAKE dataset
make train         # Train model
make serve         # Start API (dev mode)
make docker-up     # Start full stack (API + UI + MLflow)
make deploy        # Trigger Cloud Run deploy via GitHub Actions
```

## Data Management (DVC)

Dataset files are tracked with DVC. Never commit raw data to Git.

```bash
# Pull existing data
dvc pull

# After adding new data
dvc add data/processed
git add data/processed.dvc
git commit -m "data: update processed dataset"
dvc push
```

## Docker

```bash
# Build images
make docker-build

# Run full stack
make docker-up

# Check logs
make docker-logs

# Tear down
make docker-down
```

## Infrastructure (Terraform)

The `terraform/` directory provisions GCP resources using a modular structure. See [INFRASTRUCTURE.md](INFRASTRUCTURE.md) for full details.

```bash
# Choose environment
cd terraform/environments/dev   # or prod

# Configure
cp terraform.tfvars.example terraform.tfvars  # Edit with your project ID

# Deploy
terraform init
terraform plan
terraform apply
```

## Documentation

- Update `README.md` for user-facing changes
- Update `docs/ARCHITECTURE.md` for system design changes
- Add docstrings to all public functions
- Use Google-style docstrings

## Release Process

1. Update the version in `pyproject.toml` and `src/__init__.py`
2. Update `CHANGELOG.md` with the new version and changes
3. Create a release tag: `git tag v1.x.x`
4. Push the tag: `git push origin v1.x.x`
5. The CI/CD pipeline automatically builds and deploys
