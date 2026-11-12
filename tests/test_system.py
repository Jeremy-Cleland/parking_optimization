"""Deterministic end-to-end tests using the current simulation API."""

import json
from datetime import datetime

import pytest

from simulation import CitySimulator


@pytest.fixture
def simulator():
    sim = CitySimulator(
        n_zones=4,
        n_drivers=20,
        use_real_data=False,
        seed=42,
        start_time=datetime(2026, 11, 12, 8),
    )
    yield sim
    sim.coordinator.executor.shutdown(wait=True)


def test_offline_simulation_parks_drivers_and_preserves_capacity(simulator, tmp_path):
    simulator.run_simulation(duration_hours=0.5, time_step_minutes=5)
    assert simulator.metrics["successful_parks"] > 0
    assert len(simulator.drivers) == 20
    assert len(simulator.parking_zones) == 4
    parked = [driver for driver in simulator.drivers if driver["status"] == "parked"]
    assert sum(zone.occupied_spots for zone in simulator.parking_zones) == len(parked)
    assert all(
        0 <= zone.occupied_spots <= zone.capacity for zone in simulator.parking_zones
    )
    path = tmp_path / "results.json"
    simulator.save_results(str(path))
    result = json.loads(path.read_text())
    assert result["simulation_config"]["data_source"] == "synthetic"
    assert result["simulation_config"]["seed"] == 42
    assert result["metrics"]["elapsed_minutes"][-1] == 30


def test_seeded_runs_are_independent_of_global_random_state(simulator):
    import random

    import numpy as np

    simulator.run_simulation(duration_hours=0.5)
    random.seed(999)
    np.random.seed(999)
    replay = CitySimulator(
        n_zones=4,
        n_drivers=20,
        use_real_data=False,
        seed=42,
        start_time=simulator.start_time,
    )
    try:
        replay.run_simulation(duration_hours=0.5)
        assert replay.metrics == simulator.metrics
        assert [zone.current_price for zone in replay.parking_zones] == [
            zone.current_price for zone in simulator.parking_zones
        ]
        assert replay.drivers == simulator.drivers
    finally:
        replay.coordinator.executor.shutdown(wait=True)
