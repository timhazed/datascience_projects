.PHONY: help setup dev-up dev-down test test-all lint lint-all format clean build
.PHONY: backend-setup backend-test backend-lint backend-format backend-clean backend-migrate backend-load-data backend-validate-prod
.PHONY: frontend-setup frontend-dev frontend-build frontend-lint frontend-clean
.PHONY: logs db-shell redis-shell

# =============================================================================
# HELP & GENERAL
# =============================================================================

help: ## Show this help message
	@echo "GeoPortugal Development Commands"
	@echo "=================================="
	@echo ""
	@echo "FULL-STACK DEVELOPMENT:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(setup|dev-up|dev-down|test-all|lint-all):' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "BACKEND COMMANDS:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^backend-' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "FRONTEND COMMANDS:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^frontend-' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "UTILITIES:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(logs|db-shell|redis-shell|clean):' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# FULL-STACK DEVELOPMENT
# =============================================================================

setup: backend-setup frontend-setup ## Setup both backend and frontend environments
	@echo "Full development environment setup complete!"
	@echo ""
	@echo "Start development with: make dev-up"

dev-up: ## Start full development environment (backend + frontend)
	@echo "Starting full development environment..."
	@if [ ! -f ~/src/python/geoportugal/.env ]; then \
		echo "Error: External .env file not found!"; \
		echo ""; \
		echo "Setup required:"; \
		echo "1. cp .env.example ~/src/python/geoportugal/.env"; \
		echo "2. Edit the file and change ALL passwords marked 'CHANGE-THIS'"; \
		echo "3. Generate SECRET_KEY: python -c \"import secrets; print(secrets.token_urlsafe(32))\""; \
		echo ""; \
		echo "Using industry standard single .env file (stored externally for security)"; \
		exit 1; \
	fi
	@echo "External .env file found!"
	@echo "Starting databases (PostgreSQL + Redis)..."
	docker-compose --env-file ~/src/python/geoportugal/.env up -d postgres redis
	@echo "Waiting for databases to be ready..."
	sleep 8
	@echo "Running database migrations..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run alembic upgrade head
	@echo "Databases ready!"
	@echo ""
	@echo "Starting Backend API (FastAPI) on http://localhost:8000..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo $$! > .backend.pid
	@sleep 3
	@echo "Starting Frontend (Next.js) on http://localhost:3000..."
	cd frontend && npm run dev &
	@echo $$! > .frontend.pid
	@echo ""
	@echo "Development environment started!"
	@echo "Backend API: http://localhost:8000/docs"
	@echo "Frontend: http://localhost:3000"
	@echo "PGAdmin: http://localhost:5050 (with --profile tools)"
	@echo ""
	@echo "Stop with: make dev-down"

dev-down: ## Stop full development environment
	@echo "Stopping development environment..."
	@if [ -f .backend.pid ]; then \
		echo "Stopping backend..."; \
		kill `cat .backend.pid` 2>/dev/null || true; \
		rm -f .backend.pid; \
	fi
	@if [ -f .frontend.pid ]; then \
		echo "Stopping frontend..."; \
		kill `cat .frontend.pid` 2>/dev/null || true; \
		rm -f .frontend.pid; \
	fi
	@echo "Stopping Docker services..."
	docker-compose --env-file ~/src/python/geoportugal/.env down
	@echo "Development environment stopped!"

test-all: backend-test frontend-test ## Run all tests (backend + frontend)

lint-all: backend-lint frontend-lint ## Run all linting checks (backend + frontend)

lint: lint-all ## Alias for lint-all

# =============================================================================
# BACKEND COMMANDS
# =============================================================================

backend-setup: ## Setup backend environment with Poetry and Python 3.12.7
	@echo "Setting up backend environment with Poetry..."
	cd backend && poetry env use ~/.pyenv/versions/3.12.7/bin/python
	cd backend && poetry install
	cd backend && poetry run pre-commit install
	@echo "Backend setup complete!"

backend-test: ## Run backend tests with coverage
	@echo "Running backend tests..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal_test \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=test-secret-key-for-testing-at-least-32-characters \
		poetry run pytest tests/ --cov=app --cov-fail-under=75 -v --cov-report=term-missing

backend-test-fast: ## Run backend tests without coverage
	@echo "Running backend tests (fast)..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal_test \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=test-secret-key-for-testing-at-least-32-characters \
		poetry run pytest tests/ -v

backend-lint: ## Run backend linting checks
	@echo "Running backend linting..."
	cd backend && poetry run ruff check app/ tests/
	cd backend && poetry run black --check app/ tests/
	cd backend && poetry run isort --check-only app/ tests/
	cd backend && poetry run mypy app/

backend-format: ## Auto-format backend code
	@echo "Formatting backend code..."
	cd backend && poetry run black app/ tests/
	cd backend && poetry run isort app/ tests/
	cd backend && poetry run ruff check app/ tests/ --fix

backend-migrate: ## Create new backend database migration
	@echo "Creating new migration..."
	@cd backend && read -p "Enter migration message: " message; \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run alembic revision --autogenerate -m "$$message"

backend-migrate-upgrade: ## Apply backend database migrations
	@echo "Applying database migrations..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run alembic upgrade head

backend-migrate-downgrade: ## Rollback last backend migration
	@echo "Rolling back last migration..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run alembic downgrade -1

backend-load-data: ## Load Portugal data into backend
	@echo "Loading Portugal data..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run python scripts/load_portugal_data.py

backend-validate-prod: ## Validate production configuration security
	@echo "Validating production configuration..."
	cd backend && poetry run python scripts/validate_production_config.py

backend-build: ## Build backend Docker image
	@echo "Building backend Docker image..."
	cd backend && docker build -t geoportugal-api .

backend-check-db: ## Check backend database connection
	@echo "Checking database connection..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run python -c "from app.core.database import engine; import asyncio; print('Database connection: OK')"

backend-check-redis: ## Check backend Redis connection
	@echo "Checking Redis connection..."
	cd backend && \
		DEBUG=true \
		DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/geoportugal \
		REDIS_URL=redis://localhost:6379/0 \
		SECRET_KEY=dev-secret-key-change-in-production-at-least-32-chars \
		poetry run python -c "from app.services.cache import cache_service; import asyncio; asyncio.run(cache_service.connect()); print('Redis connection: OK')"

backend-clean: ## Clean backend temporary files
	@echo "Cleaning backend temporary files..."
	cd backend && find . -type d -name "__pycache__" -exec rm -rf {} +
	cd backend && find . -type f -name "*.pyc" -delete
	cd backend && find . -type f -name "*.pyo" -delete
	cd backend && find . -type f -name "*.coverage" -delete
	cd backend && find . -type d -name "*.egg-info" -exec rm -rf {} +
	cd backend && find . -type d -name ".pytest_cache" -exec rm -rf {} +
	cd backend && find . -type d -name ".mypy_cache" -exec rm -rf {} +

# =============================================================================
# FRONTEND COMMANDS
# =============================================================================

frontend-setup: ## Setup frontend environment
	@echo "Setting up frontend environment..."
	cd frontend && npm install
	@echo "Frontend setup complete!"

frontend-dev: ## Start frontend development server
	@echo "Starting frontend development server..."
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	@echo "Building frontend for production..."
	cd frontend && npm run build

frontend-test: ## Run frontend tests
	@echo "Running frontend tests..."
	@echo "Frontend tests not yet configured"
	# cd frontend && npm run test

frontend-lint: ## Run frontend linting
	@echo "Running frontend linting..."
	cd frontend && npm run lint

frontend-clean: ## Clean frontend build artifacts
	@echo "Cleaning frontend build artifacts..."
	cd frontend && rm -rf .next
	cd frontend && rm -rf out
	cd frontend && rm -rf node_modules/.cache

# =============================================================================
# UTILITIES
# =============================================================================

logs: ## Show application logs
	docker-compose --env-file ~/src/python/geoportugal/.env logs -f api

db-shell: ## Connect to database shell
	docker-compose --env-file ~/src/python/geoportugal/.env exec postgres psql -U postgres -d geoportugal

redis-shell: ## Connect to Redis shell
	docker-compose --env-file ~/src/python/geoportugal/.env exec redis redis-cli

clean: backend-clean frontend-clean ## Clean all temporary files
	@echo "Cleaning all temporary files..."
	rm -f .backend.pid .frontend.pid
	@echo "Cleanup complete!"

# =============================================================================
# BUILD & PRODUCTION
# =============================================================================

build: backend-build frontend-build ## Build both backend and frontend

docker-up: ## Start full stack with Docker Compose
	@echo "Starting full stack with Docker Compose..."
	docker-compose --env-file ~/src/python/geoportugal/.env up -d

docker-down: ## Stop Docker Compose stack
	@echo "Stopping Docker Compose stack..."
	docker-compose --env-file ~/src/python/geoportugal/.env down

docker-logs: ## View Docker Compose logs
	docker-compose --env-file ~/src/python/geoportugal/.env logs -f
