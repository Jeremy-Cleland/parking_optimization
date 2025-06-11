# Output Directory

This directory contains all simulation results, analysis reports, and generated artifacts organized by run.

## Directory Structure

```text
output/
├── runs/                      # Individual simulation runs
│   ├── run_2025-12-05_19-51-49/ # Timestamped run directories
│   │   ├── metadata.json        # Run information and artifacts list
│   │   ├── simulation_results.json  # Simulation metrics and data
│   │   ├── analysis/             # Complexity analysis and reports
│   │   │   ├── complexity_report.txt
│   │   │   └── complexity_analysis.png
│   │   ├── visualizations/       # Generated charts and dashboards
│   │   │   ├── summary_dashboard.png
│   │   │   ├── performance_metrics.png
│   │   │   └── *.png
│   │   └── logs/                # Run-specific log files
│   └── run_legacy_artifacts/     # Pre-run-management artifacts
├── latest -> runs/run_xxx/       # Symlink to most recent run
└── map_data/                     # Static real-world map data
    ├── grand_rapids_drive_network.graphml
    ├── downtown_boundary.geojson
    ├── parking_lots.geojson
    └── parking_meters.geojson
```

## Key Features

### Automatic Run Tracking

- Every simulation creates a timestamped directory
- All artifacts are automatically organized by run
- Metadata tracks parameters, duration, and status

### Latest Run Access

- `output/latest/` always points to the most recent run
- Quick access to current results without remembering timestamps

### Run Metadata

Each run directory contains `metadata.json` with:

```json
{
  "run_id": "run_2025-12-05_19-51-49",
  "timestamp": "2025-12-05T19:51:49.123456",
  "mode": "demo",
  "parameters": {
    "drivers": 50,
    "duration": 0.5,
    "zones": 20
  },
  "duration_seconds": 12.3,
  "status": "completed",
  "artifacts": ["simulation_results.json", "analysis/...", "visualizations/..."]
}
```

## Managing Runs

### List All Runs

```bash
python scripts/manage_runs.py list
```

### View Run Details

```bash
python scripts/manage_runs.py show run_2025-12-05_19-51-49
```

### Open Run Directory

```bash
python scripts/manage_runs.py open run_2025-12-05_19-51-49
```

### Compare Two Runs

```bash
python scripts/manage_runs.py compare run_1 run_2
```

### Clean Up Old Runs

```bash
python scripts/manage_runs.py cleanup --keep 10
```

## Generated Artifacts

### Simulation Results (`simulation_results.json`)

- Complete simulation metrics and performance data
- Driver behavior statistics
- Occupancy rates and revenue analysis
- Real-time optimization results

### Analysis Reports (`analysis/`)

- **complexity_report.txt**: Algorithmic complexity analysis
- **complexity_analysis.png**: Performance visualization charts

### Visualizations (`visualizations/`)

- **summary_dashboard.png**: Complete system overview
- **performance_metrics.png**: Real-time performance charts
- **occupancy_heatmap.png**: Parking zone utilization
- **revenue_analysis.png**: Dynamic pricing effectiveness
- **algorithm_comparison.png**: Algorithm performance comparison

### Logs (`logs/`)

- Detailed execution logs with timestamps
- Error tracking and debugging information
- API call monitoring and fallback usage

## Visualization Examples

The system generates comprehensive visualizations showing:

- **Real-time Performance**: Response times, throughput, success rates
- **Geographic Analysis**: Parking zone utilization heatmaps
- **Revenue Optimization**: Dynamic pricing effectiveness
- **Driver Behavior**: Search patterns and decision-making
- **Algorithm Efficiency**: Complexity analysis and benchmarks

## File Organization Benefits

1. **Easy Comparison**: Compare runs side-by-side
2. **Historical Tracking**: See how performance changes over time
3. **Reproducibility**: All parameters and results in one place
4. **Easy Cleanup**: Remove old runs without losing important data
5. **Batch Analysis**: Process multiple runs programmatically

## Quick Access

- **Latest results**: `output/latest/simulation_results.json`
- **Latest visualizations**: `output/latest/visualizations/`
- **Latest analysis**: `output/latest/analysis/`
- **Run comparison**: `python scripts/manage_runs.py compare run1 run2`
