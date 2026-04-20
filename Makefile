# SkyCop — Development Makefile
# Run `make help` to see all available targets.
#
# Conventions:
#   exp         — one-off experiments in scripts/ (discovery-based, no Makefile edit per script)
#   app         — the main SkyCop pipeline (scaffolded alongside skycop/ package)
#   test/lint   — quality gates (scaffolded with the package)
#
# The container runs as the host user (CARLA_UID/GID exported below → docker-compose.yml).

SHELL := /bin/bash
export CARLA_UID := $(shell id -u)
export CARLA_GID := $(shell id -g)

COMPOSE := docker compose
CLIENT  := $(COMPOSE) exec client

.PHONY: help setup up down clean rebuild \
        status logs logs-server \
        exp exp-list \
        app \
        shell test lint fmt \
        map-list map-load

## —— Setup & Lifecycle ——————————————————————————

help: ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## First-time setup: create .env, cache dirs, build client image
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	@mkdir -p .client-cache output scripts
	$(COMPOSE) build
	@echo ""
	@echo "Setup complete. Run 'make up' to start CARLA server + client."

up: ## Start CARLA server and client containers
	$(COMPOSE) up -d
	@echo ""
	@echo "Starting CARLA server (30-60s on first launch). Run 'make status' to check."

down: ## Stop all containers
	$(COMPOSE) down

clean: ## Stop containers and remove caches + output
	$(COMPOSE) down -v --remove-orphans 2>/dev/null || true
	rm -rf .client-cache output
	@echo "Cleaned up. Run 'make setup' to start fresh."

rebuild: ## Force rebuild client image (use after Dockerfile changes)
	$(COMPOSE) build --no-cache client

## —— Observability ————————————————————————————————

status: ## Show container status, CARLA health, and GPU
	@echo "=== Containers ==="
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "No containers running"
	@echo ""
	@echo "=== CARLA Server ==="
	@$(CLIENT) python3 -c "import carla; c=carla.Client('carla-server',2000); c.set_timeout(30); w=c.get_world(); s=w.get_settings(); print(f'  Map: {w.get_map().name}  Sync: {s.synchronous_mode}')" 2>/dev/null || echo "  Not ready (server may still be starting)"
	@echo ""
	@echo "=== GPU ==="
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

logs: ## Tail client container logs
	$(COMPOSE) logs -f client

logs-server: ## Tail CARLA server logs
	$(COMPOSE) logs -f carla-server

## —— Experiments (scripts/) ——————————————————————

exp-list: ## List all experiment scripts in scripts/
	@echo "Experiments (scripts/):"
	@if ls scripts/*.py >/dev/null 2>&1; then \
	   ls -1 scripts/*.py | sed -e 's|scripts/|  |' -e 's|\.py$$||'; \
	 else \
	   echo "  (none)"; \
	 fi

exp: ## Run an experiment. Usage: make exp N=03  -or-  make exp SCRIPT=scripts/03_foo.py
	@if [ -n "$(SCRIPT)" ]; then \
	   echo "Running $(SCRIPT) ..."; \
	   $(CLIENT) python3 "$(SCRIPT)"; \
	 elif [ -n "$(N)" ]; then \
	   match=$$(ls scripts/$(N)_*.py 2>/dev/null | head -n1); \
	   if [ -z "$$match" ]; then \
	     echo "No experiment matching scripts/$(N)_*.py"; \
	     echo "Run 'make exp-list' to see available experiments."; \
	     exit 1; \
	   fi; \
	   echo "Running $$match ..."; \
	   $(CLIENT) python3 "$$match"; \
	 else \
	   echo "Usage:"; \
	   echo "  make exp N=03                        # fuzzy-match scripts/03_*.py"; \
	   echo "  make exp SCRIPT=scripts/03_foo.py    # full path"; \
	   echo "  make exp-list                        # list available experiments"; \
	   exit 1; \
	 fi

## —— Application (skycop/) ———————————————————————

app: ## Run the SkyCop mission (python -m skycop.main)
	$(CLIENT) python3 -m skycop.main

## —— Dev —————————————————————————————————————————

shell: ## Open bash shell in client container
	$(CLIENT) bash

test: ## Run unit tests (pytest)
	$(CLIENT) python3 -m pytest

lint: ## Lint check (ruff; config in pyproject.toml)
	$(CLIENT) python3 -m ruff check .

fmt: ## Auto-format with ruff
	$(CLIENT) python3 -m ruff format .

## —— World Control ———————————————————————————————

map-list: ## List all available CARLA maps
	@$(CLIENT) python3 -c "\
		import carla, os; \
		c = carla.Client(os.environ['CARLA_HOST'], 2000); c.set_timeout(10); \
		[print(f'  {m}') for m in sorted(c.get_available_maps())]"

map-load: ## Load a CARLA map. Usage: make map-load MAP=Town03
	@test -n "$(MAP)" || (echo "Usage: make map-load MAP=Town03" && exit 1)
	@$(CLIENT) python3 -c "\
		import carla, os; \
		c = carla.Client(os.environ['CARLA_HOST'], 2000); c.set_timeout(60); \
		maps = c.get_available_maps(); \
		match = [m for m in maps if '$(MAP)' in m]; \
		match or exit('No map matching $(MAP)'); \
		print(f'Loading {match[0]}...'); \
		c.load_world(match[0]); \
		print('Done')"
