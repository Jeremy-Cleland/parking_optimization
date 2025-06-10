# Scripts Directory

Utility scripts and tools for the parking optimization system.

## Files

### Setup & Environment

- **`setup.sh`** - Environment setup script
  - Creates conda environment from `environment.yml`
  - Installs all required dependencies

### Development & Debugging

- **`debug_parking_system.py`** - System debugging and diagnostics
  - Analyzes road network connectivity
  - Tests parking zone reachability
  - Identifies driver routing issues

- **`performance_profiler.py`** - Performance monitoring and profiling
  - Benchmarks algorithm performance
  - Memory usage analysis
  - Execution timing reports

### Run Management

- **`manage_runs.py`** - Simulation run management
  - List, view, and compare simulation runs
  - Cleanup old run data
  - Export run results

### Data & Analysis

- **`calculate_map_tiles.py`** - Map tile requirements calculator
  - Estimates tile coverage for Grand Rapids data
  - Calculates storage requirements for different zoom levels

- **`fetch_grand_rapids_data.py`** - Grand Rapids map data fetcher
  - Downloads real-world map data
  - Processes parking zone information

## Usage

```bash
# Environment setup
bash scripts/setup.sh

# Debug system issues
python scripts/debug_parking_system.py

# Profile performance
python scripts/performance_profiler.py

# Manage runs
python scripts/manage_runs.py list
python scripts/manage_runs.py show <run_id>
python scripts/manage_runs.py cleanup --keep 10

# Calculate map requirements
python scripts/calculate_map_tiles.py
```

## Quick Setup

```bash
# One-command setup
make setup

# Or manually
conda env create -f environment.yml
conda activate parking_optimization
```
