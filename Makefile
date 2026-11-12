# Parking Optimization System Makefile

.PHONY: help run test setup clean report status deps install dev-tools lint format type-check pre-commit build docs

# Default target
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy
BANDIT ?= .venv/bin/bandit

# Color codes for output
GREEN := \033[32m
BLUE := \033[34m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Show available commands
	@echo "$(BLUE)Parking Optimization System - Available Commands:$(NC)"
	@echo "$(BLUE)==================================================$(NC)"
	@echo ""
	@echo "$(GREEN)🚀 Main Application:$(NC)"
	@grep -E '^(run|simulate|report):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🧪 Testing & Quality:$(NC)"
	@grep -E '^(test|test-coverage|lint|format|type-check|pre-commit):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🛠️  Development:$(NC)"
	@grep -E '^(setup|install|dev-tools|deps|build|docs):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🧹 Cleanup & Analysis:$(NC)"
	@grep -E '^(clean|clean-output|clean-artifacts|clean-all|clean-everything|list-runs|show-run|cleanup-runs|status):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# 🚀 MAIN APPLICATION COMMANDS
# =============================================================================

run: ## Run a seeded offline demo
	$(PYTHON) main.py --mode demo --seed 42

simulate: ## Run city simulation
	@echo "$(GREEN)Running city simulation...$(NC)"
	$(PYTHON) main.py --mode simulate --zones 20 --drivers 500

report: ## Generate analysis and visualization report
	@echo "$(GREEN)Generating comprehensive report...$(NC)"
	@results="$$($(PYTHON) -c 'from pathlib import Path; print(Path("output/latest/simulation_results.json").resolve())')"; \
		$(PYTHON) main.py --mode analyze && \
		$(PYTHON) main.py --mode visualize --input "$$results"
	@echo "$(BLUE)Report complete! Check output/latest/ for results$(NC)"

ensure-data: ## Ensure map data is available (download if missing)
	@echo "$(GREEN)Checking map data availability...$(NC)"
	@if [ ! -d "output/map_data" ] || [ ! -f "output/map_data/grand_rapids_drive_network.graphml" ]; then \
		echo "$(YELLOW)Map data not found. Downloading Grand Rapids data...$(NC)"; \
		$(PYTHON) scripts/fetch_grand_rapids_data.py; \
		echo "$(GREEN)✅ Map data downloaded successfully!$(NC)"; \
	else \
		echo "$(GREEN)✅ Map data already available.$(NC)"; \
	fi

fetch-data: ## Download Grand Rapids map data (force refresh)
	@echo "$(GREEN)Downloading fresh Grand Rapids map data...$(NC)"
	$(PYTHON) scripts/fetch_grand_rapids_data.py
	@echo "$(GREEN)✅ Map data download complete!$(NC)"

showcase: ## Prepare showcase directory with latest run for sharing/upload
	@echo "$(GREEN)Creating showcase directory with latest run...$(NC)"
	@mkdir -p showcase/latest_run/selected_visuals
	@mkdir -p showcase/sample_outputs
	@mkdir -p showcase/presentation
	@if [ -L "output/latest" ] && [ -d "output/latest" ]; then \
		echo "$(BLUE)Copying latest run data...$(NC)"; \
		cp output/latest/metadata.json showcase/latest_run/ 2>/dev/null || true; \
		cp output/latest/simulation_results.json showcase/latest_run/ 2>/dev/null || true; \
		echo "$(BLUE)Selecting key visualizations...$(NC)"; \
		cp output/latest/visualizations/interactive_parking_map.html showcase/latest_run/selected_visuals/ 2>/dev/null || true; \
		cp output/latest/visualizations/geographic_dashboard.png showcase/latest_run/selected_visuals/ 2>/dev/null || true; \
		cp output/latest/visualizations/performance_metrics.png showcase/latest_run/selected_visuals/ 2>/dev/null || true; \
		cp output/latest/visualizations/complexity_analysis.png showcase/latest_run/selected_visuals/ 2>/dev/null || true; \
		cp output/latest/visualizations/summary_dashboard.png showcase/latest_run/selected_visuals/ 2>/dev/null || true; \
		cp output/latest/visualizations/network_analysis_map.png showcase/latest_run/selected_visuals/ 2>/dev/null || true; \
 \
		cp output/latest/visualizations/revenue_analysis.png showcase/latest_run/selected_visuals/ 2>/dev/null || true; \
		cp output/latest/analysis/complexity_analysis.png showcase/latest_run/selected_visuals/algorithm_complexity.png 2>/dev/null || true; \
		echo "$(GREEN)✅ Showcase directory ready at: showcase/$(NC)"; \
		echo "$(BLUE)📁 Files prepared for upload/sharing:$(NC)"; \
		find showcase/latest_run -name "*.png" -o -name "*.html" -o -name "*.json" | sort; \
	else \
		echo "$(YELLOW)⚠️  No latest run found. Run 'make run' first.$(NC)"; \
	fi

# =============================================================================
# 🧪 TESTING & CODE QUALITY
# =============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running system tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ \
		--cov=core --cov=simulation --cov=analysis \
		--cov-report=html --cov-report=term --cov-report=xml

test-fast: ## Run tests without slow markers
	@echo "$(GREEN)Running fast tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "integration"

lint: ## Check code style and potential issues
	@echo "$(GREEN)Checking code style...$(NC)"
	$(RUFF) check .

lint-fix: ## Fix auto-fixable linting issues
	@echo "$(GREEN)Fixing linting issues...$(NC)"
	@$(RUFF) check . --fix

format: ## Format code with Ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	$(RUFF) format .

type-check: ## Run static type checking
	@echo "$(GREEN)Running type checks...$(NC)"
	$(MYPY) core simulation analysis main.py

security-check: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	$(BANDIT) -r core simulation analysis main.py -c pyproject.toml

pre-commit: ## Run all pre-commit checks
	@echo "$(GREEN)Running pre-commit checks...$(NC)"
	@pre-commit run --all-files || { echo "$(RED)Pre-commit not installed. Run 'make dev-tools' first$(NC)"; exit 1; }

format-check: ## Verify formatting without modifying files
	$(RUFF) format --check .

quality: ## Run all checks, including the unfinished strict typing audit
	@echo "$(GREEN)Running all quality checks...$(NC)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) test

# =============================================================================
# 🛠️  DEVELOPMENT SETUP
# =============================================================================

install: ## Install the package into the pinned virtual environment
	uv pip install --python $(PYTHON) --no-deps -e .

install-dev: ## Sync the pinned development dependencies
	uv pip sync --python $(PYTHON) requirements-dev.txt
	$(MAKE) install

dev-tools: install-dev ## Install development tools and pre-commit hooks
	.venv/bin/pre-commit install

setup: ## Create the pinned development environment
	mise exec -- python -m venv .venv
	$(MAKE) install-dev

deps: install-dev ## Restore pinned dependencies

build: ## Build the distributable wheel
	$(PYTHON) -m pip wheel --no-deps . --wheel-dir dist

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(NC)"
	@echo "$(YELLOW)Documentation generation not configured yet$(NC)"
	@echo "Add your docs generation command here"

# =============================================================================
# 🧹 CLEANUP & MAINTENANCE
# =============================================================================

clean: ## Clean temporary files and caches
	@echo "$(GREEN)Cleaning temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov build dist 2>/dev/null || true
	rm -f .coverage coverage.xml 2>/dev/null || true

clean-output: ## Clean simulation output files
	@echo "$(GREEN)Cleaning output files...$(NC)"
	rm -f output/*.json output/*.png output/*.txt 2>/dev/null || true
	rm -rf logs visualization_output 2>/dev/null || true

clean-artifacts: ## Remove all simulation artifacts and run data
	@echo "$(GREEN)Cleaning all simulation artifacts...$(NC)"
	rm -rf output/runs/* 2>/dev/null || true
	rm -rf output/latest 2>/dev/null || true
	rm -rf visualization_output/* 2>/dev/null || true
	rm -rf logs/* 2>/dev/null || true
	rm -f output/*.json output/*.png output/*.txt 2>/dev/null || true
	@echo "$(YELLOW)Preserving map data in output/map_data/$(NC)"
	@echo "$(GREEN)✅ All simulation artifacts cleared!$(NC)"

clean-all: ## Clean everything (temp files + output + caches)
	@$(MAKE) clean
	@$(MAKE) clean-output
	@$(MAKE) clean-artifacts
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

clean-everything: ## Nuclear option: clean all artifacts AND map data
	@echo "$(RED)⚠️  WARNING: This will delete ALL data including map files!$(NC)"
	@echo "$(YELLOW)Press Ctrl+C in the next 5 seconds to cancel...$(NC)"
	@sleep 5
	@echo "$(GREEN)Proceeding with complete cleanup...$(NC)"
	@$(MAKE) clean
	@$(MAKE) clean-output
	rm -rf output/ 2>/dev/null || true
	rm -rf visualization_output/ 2>/dev/null || true
	rm -rf logs/ 2>/dev/null || true
	rm -rf cache/ 2>/dev/null || true
	@echo "$(GREEN)✅ Everything cleaned! Next run will download fresh data.$(NC)"

# =============================================================================
# 📊 ANALYSIS & MONITORING
# =============================================================================

list-runs: ## List all simulation runs
	$(PYTHON) scripts/manage_runs.py list

show-run: ## Show the latest run's details
	$(PYTHON) scripts/manage_runs.py show "$$(basename "$$(readlink output/latest)")"

cleanup-runs: ## Clean up old runs (keep 0)
	$(PYTHON) scripts/manage_runs.py cleanup --keep 0

status: ## Show comprehensive project status
	@echo "$(BLUE)Parking Optimization Project Status:$(NC)"
	@echo "$(BLUE)====================================$(NC)"
	@echo "$(GREEN)Environment:$(NC)"
	@echo "  Directory: $(PWD)"
	@echo "  Python: $$(which python 2>/dev/null || echo 'Not found')"
	@echo "  Conda Env: $$(conda info --envs 2>/dev/null | grep '*' | awk '{print $$1}' || echo 'Not in conda')"
	@echo ""
	@echo "$(GREEN)Development Tools:$(NC)"
	@echo "  Ruff: $$(which ruff >/dev/null 2>&1 && echo '✅ Installed' || echo '❌ Missing')"
	@echo "  MyPy: $$(which mypy >/dev/null 2>&1 && echo '✅ Installed' || echo '❌ Missing')"
	@echo "  Pytest: $$(which pytest >/dev/null 2>&1 && echo '✅ Installed' || echo '❌ Missing')"
	@echo "  Pre-commit: $$(which pre-commit >/dev/null 2>&1 && echo '✅ Installed' || echo '❌ Missing')"
	@echo ""
	@echo "$(GREEN)API Configuration:$(NC)"
	@echo "  Map Provider: $$(echo $${MAP_PROVIDER:-tomtom})"
	@echo "  TomTom API: $$([ -n "$$TOMTOM_API_KEY" ] && echo '✅ Set' || echo '❌ Not set')"
	@echo "  Google Maps API: $$([ -n "$$GOOGLE_MAPS_API_KEY" ] && echo '✅ Set' || echo '❌ Not set')"
	@echo "  Mapbox Token: $$([ -n "$$MAPBOX_ACCESS_TOKEN" ] && echo '✅ Set' || echo '❌ Not set')"
	@echo ""
	@echo "$(GREEN)Project Data:$(NC)"
	@echo "  Simulation Runs: $$(ls -1 output/runs/ 2>/dev/null | wc -l | tr -d ' ')"
	@echo "  Latest Run: $$(ls -la output/latest 2>/dev/null | awk '{print $$NF}' || echo 'None')"
	@echo "  Test Coverage: $$([ -f htmlcov/index.html ] && echo '✅ Available (htmlcov/)' || echo '❌ Run make test-coverage')"

# =============================================================================
# 🚦 WORKFLOW SHORTCUTS
# =============================================================================

dev-setup: ## Complete development setup from scratch
	@echo "$(GREEN)Setting up complete development environment...$(NC)"
	@$(MAKE) install-dev
	@$(MAKE) dev-tools
	@echo "$(BLUE)Development setup complete! You can now run:$(NC)"
	@echo "  $(YELLOW)make quality$(NC)  - Run all quality checks"
	@echo "  $(YELLOW)make run$(NC)      - Run the application"
	@echo "  $(YELLOW)make test$(NC)     - Run tests"

ci: ## Run the checks enforced by GitHub CI (typing audit remains separate)
	$(MAKE) lint format-check security-check test build
	$(PYTHON) -m scripts.export_evaluation --verify

demo: ## Run a quick, seeded offline demo
	$(PYTHON) main.py --mode demo
