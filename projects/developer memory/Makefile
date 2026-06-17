# Developer Memory — Makefile
#
# Usage:
#   make up          — build images and start all services (native Ollama required)
#   make ui          — open Streamlit UI in browser
#   make down        — stop and remove containers
#   make logs        — tail logs from all containers
#   make status      — show container health
#   make dump        — down + remove ChromaDB volume (resync required to rebuild index)
#   make clean       — down + remove volumes (WARNING: destroys ChromaDB data)
#   make test        — run pytest suite
#   make lint        — run ruff check
#
# Prerequisites:
#   1. Docker Desktop running
#   2. Ollama running natively: `ollama serve`
#   3. Model pulled: `ollama pull gemma4:26b`
#   4. .env file created from .env.example (GITHUB_TOKEN set)
#   5. OLLAMA_ORIGINS set on macOS host:
#        launchctl setenv OLLAMA_ORIGINS "*"
#        (then restart Ollama: pkill ollama && ollama serve)

COMPOSE       := docker compose
STREAMLIT_URL := http://localhost:8501
MCP_HEALTH    := http://localhost:9000/health

.DEFAULT_GOAL := help

# ── Environment guard ─────────────────────────────────────────────────────────

.env:
	@echo "ERROR: .env file not found. Copy .env.example and fill in GITHUB_TOKEN."
	@echo "  cp .env.example .env"
	@exit 1

_check_ollama:
	@curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 || \
	  (echo "ERROR: Ollama is not running. Start it with: ollama serve" && exit 1)
	@echo "Ollama: OK"

# ── Primary targets ───────────────────────────────────────────────────────────

.PHONY: up
up: .env _check_ollama
	$(COMPOSE) up --build -d
	@echo ""
	@echo "Waiting for services to become healthy..."
	@$(MAKE) --no-print-directory _wait_healthy
	@echo ""
	@echo "Stack ready:"
	@echo "  Streamlit UI  → $(STREAMLIT_URL)"
	@echo "  MCP Server    → http://localhost:9000"
	@echo ""
	@echo "Run 'make ui' to open the UI in your browser."

.PHONY: up-fg
up-fg: .env _check_ollama
	$(COMPOSE) up --build

.PHONY: down
down:
	$(COMPOSE) down

.PHONY: restart
restart:
	$(COMPOSE) restart

.PHONY: ui
ui:
	@open $(STREAMLIT_URL) 2>/dev/null || xdg-open $(STREAMLIT_URL) 2>/dev/null || \
	  echo "Open your browser to $(STREAMLIT_URL)"

.PHONY: logs
logs:
	$(COMPOSE) logs -f

.PHONY: logs-mcp
logs-mcp:
	$(COMPOSE) logs -f mcp-server

.PHONY: logs-ui
logs-ui:
	$(COMPOSE) logs -f streamlit

.PHONY: logs-chroma
logs-chroma:
	$(COMPOSE) logs -f chromadb

.PHONY: status
status:
	$(COMPOSE) ps

# ── Health checks ─────────────────────────────────────────────────────────────

.PHONY: health
health:
	@echo "Checking service health..."
	@curl -sf $(MCP_HEALTH)    > /dev/null 2>&1 && echo "  MCP Server : healthy" || echo "  MCP Server : UNHEALTHY"
	@curl -sf $(STREAMLIT_URL)/_stcore/health > /dev/null 2>&1 && echo "  Streamlit  : healthy" || echo "  Streamlit  : UNHEALTHY"
	@docker exec developer_memory_chroma curl -sf http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1 && echo "  ChromaDB   : healthy" || echo "  ChromaDB   : UNHEALTHY"
	@curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && echo "  Ollama     : healthy" || echo "  Ollama     : UNHEALTHY (start with: ollama serve)"

.PHONY: _wait_healthy
_wait_healthy:
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12; do \
	  mcp=$$(curl -sf $(MCP_HEALTH) > /dev/null 2>&1 && echo ok || echo no); \
	  ui=$$(curl -sf $(STREAMLIT_URL)/_stcore/health > /dev/null 2>&1 && echo ok || echo no); \
	  if [ "$$mcp" = "ok" ] && [ "$$ui" = "ok" ]; then \
	    echo "  All services healthy."; \
	    exit 0; \
	  fi; \
	  echo "  Waiting... (attempt $$i/12)"; \
	  sleep 5; \
	done; \
	echo "WARNING: Some services may not be healthy yet. Check with 'make health'."

# ── Build targets ─────────────────────────────────────────────────────────────

.PHONY: build
build: .env
	$(COMPOSE) build

.PHONY: build-no-cache
build-no-cache: .env
	$(COMPOSE) build --no-cache

# ── Data management ───────────────────────────────────────────────────────────

.PHONY: clean
clean:
	@echo "WARNING: This will destroy all ChromaDB vector data."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(COMPOSE) down -v
	@echo "Volumes removed. All indexed data has been deleted."

.PHONY: dump
dump:
	@echo "Stopping containers and wiping ChromaDB + SHA store (full resync required)."
	$(COMPOSE) down
	docker volume rm developermemory_chroma_data 2>/dev/null || true
	docker volume rm developermemory_sha_store_data 2>/dev/null || true
	@echo "Done. Run 'make up' then resync repositories to rebuild the index."

.PHONY: reset-chroma
reset-chroma:
	@echo "Resetting ChromaDB volume only (MCP server and Streamlit data preserved)."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(COMPOSE) stop chromadb mcp-server
	$(COMPOSE) rm -f chromadb
	docker volume rm $$(docker volume ls -q | grep chroma_data) 2>/dev/null || true
	$(COMPOSE) up -d chromadb mcp-server
	@echo "ChromaDB reset. Re-sync repositories to rebuild the index."

# ── Development targets ───────────────────────────────────────────────────────

.PHONY: test
test:
	poetry run pytest

.PHONY: test-cov
test-cov:
	poetry run pytest --cov=src --cov-report=term-missing

.PHONY: lint
lint:
	poetry run ruff check src/ tests/ experiments/ ui/

.PHONY: lint-fix
lint-fix:
	poetry run ruff check --fix src/ tests/ experiments/ ui/

.PHONY: shell-mcp
shell-mcp:
	$(COMPOSE) exec mcp-server /bin/bash

.PHONY: shell-chroma
shell-chroma:
	$(COMPOSE) exec chromadb /bin/sh

# ── Ollama helpers ────────────────────────────────────────────────────────────

.PHONY: pull-model
pull-model:
	ollama pull gemma4:26b

.PHONY: pull-model-e4b
pull-model-e4b:
	ollama pull gemma4:e4b

.PHONY: list-models
list-models:
	ollama list

# ── Help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Developer Memory — Makefile targets"
	@echo ""
	@echo "Stack management:"
	@echo "  make up              Build images and start all services (detached)"
	@echo "  make up-fg           Start in foreground (shows live logs)"
	@echo "  make down            Stop and remove containers"
	@echo "  make restart         Restart all containers"
	@echo "  make status          Show container status"
	@echo "  make health          Check health of all four services"
	@echo "  make ui              Open Streamlit UI in browser"
	@echo ""
	@echo "Logs:"
	@echo "  make logs            Tail all service logs"
	@echo "  make logs-mcp        Tail MCP server logs only"
	@echo "  make logs-ui         Tail Streamlit logs only"
	@echo "  make logs-chroma     Tail ChromaDB logs only"
	@echo ""
	@echo "Build:"
	@echo "  make build           Build images (uses cache)"
	@echo "  make build-no-cache  Force full rebuild"
	@echo ""
	@echo "Data:"
	@echo "  make dump            Stop containers and wipe ChromaDB volume (resync required)"
	@echo "  make clean           Stop containers and delete ALL volumes (irreversible)"
	@echo "  make reset-chroma    Delete ChromaDB volume only and restart"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run pytest suite"
	@echo "  make test-cov        Run pytest with coverage report"
	@echo "  make lint            Run ruff check"
	@echo "  make lint-fix        Run ruff check --fix"
	@echo "  make shell-mcp       Open shell inside mcp-server container"
	@echo "  make shell-chroma    Open shell inside chromadb container"
	@echo ""
	@echo "Ollama:"
	@echo "  make pull-model      Pull gemma4:26b (recommended)"
	@echo "  make pull-model-e4b  Pull gemma4:e4b (quantized fallback)"
	@echo "  make list-models     List locally available Ollama models"
	@echo ""
	@echo "Prerequisites for 'make up':"
	@echo "  1. Docker Desktop running"
	@echo "  2. Ollama running natively: ollama serve"
	@echo "  3. Model pulled: make pull-model"
	@echo "  4. .env file created from .env.example"
	@echo "  5. OLLAMA_ORIGINS set: launchctl setenv OLLAMA_ORIGINS \"*\""
