# Parking Optimization System Makefile

.PHONY: help run test setup clean report status deps install dev-tools lint format type-check pre-commit build docs

# Default target
.DEFAULT_GOAL := help

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
	@grep -E '^(clean|clean-output|list-runs|show-run|cleanup-runs|status):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# 🚀 MAIN APPLICATION COMMANDS
# =============================================================================

run: ## Run simulation demo
	@echo "$(GREEN)Running parking optimization demo...$(NC)"
	python main.py --mode demo

simulate: ## Run city simulation
	@echo "$(GREEN)Running city simulation...$(NC)"
	python main.py --mode simulate --zones 20 --drivers 500

report: ## Generate analysis and visualization report
	@echo "$(GREEN)Generating comprehensive report...$(NC)"
	python main.py --mode analyze
	python main.py --mode visualize
	@echo "$(BLUE)Report complete! Check output/latest/ for results$(NC)"

# =============================================================================
# 🧪 TESTING & CODE QUALITY
# =============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running system tests...$(NC)"
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	python -m pytest tests/ \
		--cov=core --cov=simulation --cov=analysis \
		--cov-report=html --cov-report=term --cov-report=xml

test-fast: ## Run tests without slow markers
	@echo "$(GREEN)Running fast tests...$(NC)"
	python -m pytest tests/ -v -m "not slow"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	python -m pytest tests/ -v -m "integration"

lint: ## Check code style and potential issues
	@echo "$(GREEN)Checking code style...$(NC)"
	@ruff check . || { echo "$(RED)Ruff not available in environment$(NC)"; exit 1; }

lint-fix: ## Fix auto-fixable linting issues
	@echo "$(GREEN)Fixing linting issues...$(NC)"
	@ruff check . --fix

format: ## Format code with Ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	@ruff format . || { echo "$(RED)Ruff not available in environment$(NC)"; exit 1; }

type-check: ## Run static type checking
	@echo "$(GREEN)Running type checks...$(NC)"
	@mypy . || { echo "$(RED)MyPy not available in environment$(NC)"; exit 1; }

security-check: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	@bandit -r . -c pyproject.toml || { echo "$(RED)Bandit not available in environment$(NC)"; exit 1; }

pre-commit: ## Run all pre-commit checks
	@echo "$(GREEN)Running pre-commit checks...$(NC)"
	@pre-commit run --all-files || { echo "$(RED)Pre-commit not installed. Run 'make dev-tools' first$(NC)"; exit 1; }

quality: ## Run all quality checks (lint, format, type-check, test)
	@echo "$(GREEN)Running all quality checks...$(NC)"
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) test

# =============================================================================
# 🛠️  DEVELOPMENT SETUP
# =============================================================================

install: ## Install package in development mode
	@echo "$(GREEN)Installing package in development mode...$(NC)"
	conda activate parking_optimization && pip install -e .

install-dev: ## Install with development dependencies
	@echo "$(GREEN)Installing with development dependencies...$(NC)"
	conda activate parking_optimization && pip install -e ".[dev]"

dev-tools: ## Install and setup all development tools
	@echo "$(GREEN)Setting up development tools...$(NC)"
	conda activate parking_optimization && pip install -e ".[dev]"
	@echo "$(GREEN)Installing pre-commit hooks...$(NC)"
	@pre-commit install || pip install pre-commit && pre-commit install
	@echo "$(BLUE)Development tools setup complete!$(NC)"

setup: ## Set up environment and dependencies (legacy)
	@echo "$(GREEN)Setting up environment...$(NC)"
	@if [ -f environment.yml ]; then \
		conda env update -f environment.yml; \
	else \
		echo "$(YELLOW)No environment.yml found, using pyproject.toml setup$(NC)"; \
		$(MAKE) install-dev; \
	fi

deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	@if [ -f environment.yml ]; then \
		conda env update -f environment.yml; \
	else \
		conda activate parking_optimization && pip install -e ".[dev]" --upgrade; \
	fi

build: ## Build package for distribution
	@echo "$(GREEN)Building package...$(NC)"
	conda activate parking_optimization && python -m build

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

clean-all: ## Clean everything (temp files + output + caches)
	@$(MAKE) clean
	@$(MAKE) clean-output
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

# =============================================================================
# 📊 ANALYSIS & MONITORING
# =============================================================================

list-runs: ## List all simulation runs
	@echo "$(GREEN)Listing simulation runs...$(NC)"
	@if [ -f scripts/manage_runs.py ]; then \
		conda activate parking_optimization && python scripts/manage_runs.py list; \
	else \
		echo "$(YELLOW)No manage_runs.py script found$(NC)"; \
		ls -la output/runs/ 2>/dev/null || echo "$(RED)No runs directory$(NC)"; \
	fi

show-run: ## Show latest run details
	@echo "$(GREEN)Showing latest run...$(NC)"
	@if [ -f scripts/manage_runs.py ]; then \
		conda activate parking_optimization && python scripts/manage_runs.py list | grep -E "run_[0-9]" | head -1 | awk '{print $$2}' | xargs -I {} python scripts/manage_runs.py show {}; \
	else \
		echo "$(YELLOW)No manage_runs.py script found$(NC)"; \
		ls -la output/latest 2>/dev/null || echo "$(RED)No latest run$(NC)"; \
	fi

cleanup-runs: ## Clean up old runs (keep 5 most recent)
	@echo "$(GREEN)Cleaning up old runs...$(NC)"
	@if [ -f scripts/manage_runs.py ]; then \
		conda activate parking_optimization && python scripts/manage_runs.py cleanup --keep 5; \
	else \
		echo "$(YELLOW)No manage_runs.py script found$(NC)"; \
	fi

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

ci: ## Run continuous integration checks
	@echo "$(GREEN)Running CI pipeline...$(NC)"
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) security-check
	@$(MAKE) test-coverage
	@echo "$(GREEN)✅ CI pipeline passed!$(NC)"

demo: ## Quick demo run with all checks
	@echo "$(GREEN)Running quick demo with quality checks...$(NC)"
	@$(MAKE) format
	@$(MAKE) lint-fix
	@$(MAKE) test-fast
	@$(MAKE) run
	@echo "$(GREEN)✅ Demo complete!$(NC)"
