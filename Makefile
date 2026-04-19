# CARLA Testing — Development Makefile
# Run `make help` to see all available targets.

SHELL := /bin/bash
export CARLA_UID := $(shell id -u)
export CARLA_GID := $(shell id -g)

COMPOSE := docker compose
CLIENT  := $(COMPOSE) exec client

.PHONY: help setup up down clean rebuild \
        status logs logs-server shell \
        run hello-world \
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
	@echo "Starting CARLA server (this takes 30-60s on first launch)..."
	@echo "  Server: carla-server (ports 2000-2001)"
	@echo "  Client: run 'make shell' to interact, or 'make run SCRIPT=scripts/01_hello_world.py'"
	@echo ""
	@echo "Run 'make status' to check readiness."

down: ## Stop all containers
	$(COMPOSE) down

clean: ## Stop containers and remove caches + output
	$(COMPOSE) down -v --remove-orphans 2>/dev/null || true
	rm -rf .client-cache output
	@echo "Cleaned up. Run 'make setup' to start fresh."

rebuild: ## Force rebuild client image (use after Dockerfile changes)
	$(COMPOSE) build --no-cache client

## —— Server & Status ————————————————————————————

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

## —— Development —————————————————————————————————

shell: ## Open bash shell in client container
	$(CLIENT) bash

run: ## Run a Python script in client. Usage: make run SCRIPT=scripts/01_hello_world.py
	@test -n "$(SCRIPT)" || (echo "Usage: make run SCRIPT=scripts/<name>.py" && exit 1)
	$(CLIENT) python3 $(SCRIPT)

## —— World Control ——————————————————————————————

map-list: ## List all available maps
	@$(CLIENT) python3 -c "\
		import carla, os; \
		c = carla.Client(os.environ['CARLA_HOST'], 2000); c.set_timeout(10); \
		[print(f'  {m}') for m in sorted(c.get_available_maps())]"

map-load: ## Load a map. Usage: make map-load MAP=Town03
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

## —— Lessons / Experiments ———————————————————————

hello-world: ## Lesson 01: connect, spawn car, capture frame
	$(CLIENT) python3 scripts/01_hello_world.py

drone: ## Lesson 02: fly around as a drone — open http://localhost:5000
	$(CLIENT) python3 scripts/02_drone_view.py
