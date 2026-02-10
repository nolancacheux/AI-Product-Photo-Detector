# Makefile for AI Product Photo Detector
# Run `make help` for available commands

.PHONY: help install dev lint format test train serve ui docker-build docker-run docker-up docker-down predict predict-batch clean

# Default target
help:
	@echo "AI Product Photo Detector - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make lint        Run linting (ruff + mypy)"
	@echo "  make format      Format code with ruff"
	@echo "  make test        Run tests with coverage"
	@echo ""
	@echo "Run:"
	@echo "  make train       Train the model"
	@echo "  make serve       Start API server"
	@echo "  make ui          Start Streamlit UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker images"
	@echo "  make docker-run    Run API container locally"
	@echo "  make docker-up     Start all services"
	@echo "  make docker-down   Stop all services"
	@echo ""
	@echo "Testing API:"
	@echo "  make predict       Test single prediction (requires running API)"
	@echo "  make predict-batch Test batch prediction (requires running API)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       Remove build artifacts"

# Setup
install:
	uv pip install -e . || pip install -e .

dev:
	uv pip install -e ".[dev,ui]" || pip install -e ".[dev,ui]"
	pre-commit install

# Development
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/
	ruff check src/ tests/ --fix

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run
train:
	python -m src.training.train --config configs/train_config.yaml

serve:
	uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/ui/app.py --server.port 8501

# Docker
docker-build:
	docker build -f docker/train.Dockerfile -t ai-product-detector-train:1.0.0 .
	docker build -f docker/serve.Dockerfile -t ai-product-detector:1.0.0 .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models ai-product-detector:1.0.0

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Quick prediction test (requires running API on localhost:8000)
predict:
	@echo "Testing prediction with sample image..."
	curl -s -X POST http://localhost:8000/predict \
		-F "file=@tests/data/sample_real.jpg" | python -m json.tool

predict-batch:
	@echo "Testing batch prediction..."
	curl -s -X POST http://localhost:8000/predict/batch \
		-F "files=@tests/data/sample_real.jpg" \
		-F "files=@tests/data/sample_ai.png" | python -m json.tool

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
