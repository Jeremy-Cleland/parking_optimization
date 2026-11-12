"""Check optional GPU math, cache correctness, and warm-up measurement cohorts."""

from datetime import datetime

import numpy as np
import pytest

from core.array_backend import mlx_available, walking_distances
from core.route_optimizer import RouteOptimizer
from simulation.experiments import ExperimentConfig, run_experiment, summarize


def test_batched_distances_match_scalar_geography_and_handle_empty_inputs():
    destinations = np.array([[42.963, -85.668], [42.961, -85.662], [42.970, -85.674]])
    zones = np.array([[42.963, -85.668], [42.956, -85.680]])
    actual, metadata = walking_distances(destinations, zones, "numpy")
    expected = np.array(
        [
            [RouteOptimizer._distance_meters(zone, point) for zone in zones]
            for point in destinations
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=1e-5)
    assert metadata["device"] == "CPU"
    assert walking_distances([], zones, "numpy")[0].shape == (0, 2)
    with pytest.raises(ValueError):
        walking_distances(destinations, zones, "unknown")


@pytest.mark.skipif(
    not mlx_available(), reason="Optional MLX Metal backend unavailable"
)
def test_mlx_gpu_matches_numpy_for_local_geographic_batch():
    rng = np.random.default_rng(42)
    destinations = rng.uniform([42.95, -85.69], [42.98, -85.65], size=(2000, 2))
    zones = rng.uniform([42.95, -85.69], [42.98, -85.65], size=(113, 2))
    cpu, _ = walking_distances(destinations, zones, "numpy")
    gpu, metadata = walking_distances(destinations, zones, "mlx")
    np.testing.assert_allclose(gpu, cpu, atol=0.02, rtol=1e-5)
    assert metadata["backend"] == "mlx"
    assert "Apple" in metadata["device"]


def test_warmup_excludes_revenue_and_uses_post_warmup_arrival_cohort():
    results = {
        "simulation_config": {
            "warmup_hours": 1 / 6,
            "duration_hours": 0.25,
            "price_constraints": {"target_occupancy": 0.8},
        },
        "driver_outcomes": {"total": 3, "parked": 2, "rejected": 0, "unresolved": 1},
        "driver_records": [
            {
                "id": "warm",
                "arrival_minutes": 0,
                "parked_at_minutes": 5,
                "status": "parked",
                "parking_duration_minutes": 60,
            },
            {
                "id": "measured",
                "arrival_minutes": 10,
                "parked_at_minutes": 12,
                "status": "parked",
                "parking_duration_minutes": 60,
            },
            {
                "id": "late",
                "arrival_minutes": 14,
                "parked_at_minutes": None,
                "status": "searching",
                "parking_duration_minutes": 60,
            },
        ],
        "final_state": {"zone_occupancy": [{"id": "z", "capacity": 10}]},
        "metrics": {
            "elapsed_minutes": [10, 15],
            "avg_occupancy": [0.2, 0.8],
            "total_revenue": [2, 1],
            "zone_prices": [{"z": 2}, {"z": 8}],
            "parking_observations": [
                {
                    "driver_id": driver,
                    "search_minutes": minutes,
                    "driving_minutes": 1,
                    "walking_meters": 50,
                    "mean_stay_cost": 4,
                }
                for driver, minutes in [("warm", 5), ("measured", 2)]
            ],
        },
    }
    summary = summarize(results)
    assert summary["revenue"] == 1
    assert summary["measurement_hours"] == pytest.approx(1 / 12)
    assert summary["mean_occupancy"] == pytest.approx(0.8)
    assert summary["total"] == 2
    assert summary["parked"] == 1
    assert summary["unresolved"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["search_n"] == 1
    assert summary["search_minutes"] == 2


def test_parallel_experiments_preserve_seed_inputs_and_results(tmp_path):
    config = ExperimentConfig(
        seeds=(1, 2),
        policies=("static-distance", "dynamic-time"),
        n_zones=2,
        n_drivers=6,
        duration_hours=0.3,
        warmup_hours=1 / 12,
        scenario_event="steady",
        array_backend="numpy",
    )
    serial = run_experiment(config, tmp_path / "serial")
    parallel = run_experiment(config, tmp_path / "parallel", workers=2)
    assert serial == parallel


def test_warmup_must_align_and_leave_a_measurement_window():
    for kwargs in [
        {"warmup_hours": 2, "duration_hours": 2},
        {"warmup_hours": -1},
        {"warmup_hours": 0.1, "time_step_minutes": 5},
    ]:
        with pytest.raises(ValueError):
            ExperimentConfig(**kwargs)
