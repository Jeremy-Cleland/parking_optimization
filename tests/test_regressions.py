"""Regression coverage for route, clock, accounting, and CLI contracts."""

import json
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pytest

import main
from core import (
    CityCoordinator,
    DynamicPricingEngine,
    MapDataLoader,
    ParkingZone,
    RouteOptimizer,
)
from core.run_manager import RunManager
from simulation import CitySimulator


@pytest.fixture
def route_loader(tmp_path):
    loader = MapDataLoader(str(tmp_path))
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(0, y=42.963, x=-85.668)
    graph.add_node(1, y=42.964, x=-85.668)
    graph.add_node(2, y=42.965, x=-85.668)
    graph.add_edge(0, 1, length=200, travel_time=10)
    graph.add_edge(0, 1, length=100, travel_time=20)
    graph.add_edge(1, 2, length=50, speed_kph=18)
    loader.road_network = graph
    loader._parking_zones = [
        {"id": "zone", "coordinates": (42.965, -85.668), "hourly_rate": 2}
    ]
    return loader


def test_one_way_destination_is_reachable(route_loader):
    router = RouteOptimizer(map_loader=route_loader)
    options = router.find_optimal_parking((42.963, -85.668))
    assert options["zone"].path_nodes == [0, 1, 2]
    assert options["zone"].total_distance == 150
    assert options["zone"].total_time == 30
    assert (
        router.find_optimal_parking((42.965, -85.668), available_parking_zones=[]) == {}
    )


def test_multigraph_metrics_use_selected_edge_and_existing_time(route_loader):
    router = RouteOptimizer(map_loader=route_loader)
    route = router.find_optimal_route(
        (42.963, -85.668), preferences={"prefer_time": True}
    )
    assert route.total_distance == 250
    assert route.total_time == 20


def test_missing_time_weights_are_derived_before_shortest_path(route_loader):
    graph = route_loader.road_network
    graph.add_edge(0, 2, length=1000, speed_kph=[3.6])
    router = RouteOptimizer(map_loader=route_loader)
    route = router.find_optimal_route(
        (42.963, -85.668), preferences={"prefer_time": True}
    )
    assert route.path_nodes == [0, 1, 2]
    assert graph[0][2][0]["travel_time"] == 1000


@pytest.fixture
def simulator():
    sim = CitySimulator(
        n_zones=2,
        n_drivers=0,
        use_real_data=False,
        seed=42,
        start_time=datetime(2026, 11, 12, 23, 50),
    )
    yield sim
    sim.coordinator.executor.shutdown(wait=True)


def test_revenue_uses_actual_step_and_final_partial_interval(simulator, monkeypatch):
    zone = simulator.parking_zones[0]
    zone.current_price = 6
    zone.occupy_spot("existing")
    monkeypatch.setattr(simulator.coordinator, "optimize_city_parking", lambda now: {})
    simulator.run_simulation(duration_hours=0.25, time_step_minutes=10)
    assert simulator.current_time == simulator.start_time + timedelta(minutes=15)
    assert simulator.metrics["total_revenue"] == [1, 0.5]
    assert simulator.metrics["elapsed_minutes"] == [10, 15]


def test_arrival_comparison_crosses_midnight(simulator):
    driver = {
        "id": "later",
        "status": "traveling",
        "arrival_time": datetime(2026, 11, 13, 0, 5),
    }
    simulator.drivers = [driver]
    simulator._process_drivers({})
    assert driver["status"] == "traveling"
    simulator.current_time = datetime(2026, 11, 13, 0, 5)
    simulator._process_drivers({})
    assert driver["status"] == "searching"


def test_departure_releases_the_correct_vehicle(simulator):
    zone = simulator.parking_zones[0]
    zone.occupy_spot("first")
    zone.occupy_spot("second")
    driver = {"id": "second", "assigned_zone": zone.id, "status": "parked"}
    simulator._driver_departs(driver)
    assert [spot.vehicle_id for spot in zone.spots if spot.is_occupied] == ["first"]
    assert driver["status"] == "departed"


@pytest.mark.parametrize(
    "duration,step",
    [(0, 5), (-1, 5), (1, 0), (1, -5), (float("nan"), 5), (1, float("inf"))],
)
def test_invalid_time_parameters_fail_before_running(simulator, duration, step):
    with pytest.raises(ValueError):
        simulator.run_simulation(duration, step)
    assert simulator.drivers == []


def test_save_results_accepts_bare_filename(simulator, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    simulator.save_results("results.json")
    assert (
        json.loads((tmp_path / "results.json").read_text())["simulation_config"]["seed"]
        == 42
    )


def test_pricing_uses_supplied_simulation_clock():
    zone = ParkingZone("zone", "Zone", (42.963, -85.668), 10, 3, zone_type="commercial")
    pricing = DynamicPricingEngine()
    morning = pricing.calculate_zone_price(
        zone, [], current_time=datetime(2026, 11, 12, 8)
    )
    night = pricing.calculate_zone_price(
        zone, [], current_time=datetime(2026, 11, 12, 23)
    )
    assert morning > night


def test_small_city_has_no_empty_districts_and_repartition_clears_state(simulator):
    coordinator = simulator.coordinator
    assert len(coordinator.districts) == 2
    assert all(
        district.zones and np.isfinite(district.center).all()
        for district in coordinator.districts.values()
    )
    coordinator.divide_city_into_districts(simulator.parking_zones[:1])
    assert len(coordinator.districts) == 1
    assert len(coordinator.demand_predictors) == 1


def test_border_price_adjustments_are_applied(simulator):
    coordinator = simulator.coordinator
    district = next(iter(coordinator.districts.values()))
    zone = district.zones[0]
    zone.current_price = 4
    results = {district.id: {"zone_prices": {zone.id: 4}}}
    final = coordinator._apply_global_optimization(
        results, {"border_adjustments": {district.id: 1.1}}
    )
    assert final["updated_prices"][zone.id] == pytest.approx(4.4)
    assert zone.current_price == pytest.approx(4.4)
    assert results[district.id]["zone_prices"][zone.id] == pytest.approx(4.4)


@pytest.fixture
def cli_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manager = RunManager(str(tmp_path / "runs-output"))
    monkeypatch.setattr(main, "start_run", manager.start_run)
    monkeypatch.setattr(main, "get_run_path", manager.get_run_path)
    monkeypatch.setattr(main, "complete_run", manager.complete_run)
    return manager, tmp_path


def test_demo_executes_and_honors_cli_parameters(cli_run):
    manager, directory = cli_run
    assert (
        main.main(
            [
                "--zones",
                "3",
                "--drivers",
                "10",
                "--duration",
                "0.25",
                "--seed",
                "7",
                "--start-time",
                "2026-11-12T08:00:00",
                "--output",
                "results.json",
            ]
        )
        == 0
    )
    results = json.loads((directory / "results.json").read_text())
    assert results["simulation_config"]["total_zones"] == 3
    assert results["simulation_config"]["n_drivers"] == 10
    assert results["simulation_config"]["seed"] == 7
    assert results["simulation_config"]["data_source"] == "synthetic"
    assert results["metrics"]["elapsed_minutes"][-1] == 15
    assert manager.list_runs()[0]["status"] == "completed"


def test_postprocessing_failure_returns_nonzero_and_records_partial(
    cli_run, monkeypatch
):
    manager, _ = cli_run

    def fail(_directory):
        raise RuntimeError("deliberate analysis failure")

    monkeypatch.setattr(main, "generate_analysis", fail)
    assert (
        main.main(
            [
                "--mode",
                "simulate",
                "--data-source",
                "synthetic",
                "--zones",
                "1",
                "--drivers",
                "0",
                "--duration",
                "0.1",
                "--skip-visualizations",
            ]
        )
        == 1
    )
    assert manager.list_runs()[0]["status"] == "partial"


def test_visualize_missing_input_fails(cli_run):
    manager, _ = cli_run
    assert main.main(["--mode", "visualize", "--input", "missing.json"]) == 1
    assert manager.list_runs()[0]["status"] == "failed"


@pytest.mark.parametrize(
    "option,value",
    [
        ("--zones", "0"),
        ("--drivers", "-1"),
        ("--duration", "nan"),
        ("--time-step", "0"),
        ("--seed", "-2"),
    ],
)
def test_invalid_cli_inputs_fail_before_starting_run(cli_run, option, value):
    manager, _ = cli_run
    with pytest.raises(SystemExit) as error:
        main.main([option, value])
    assert error.value.code == 2
    assert manager.list_runs() == []


def test_small_synthetic_city_spawns_on_its_connected_grid(simulator):
    simulator.n_drivers = 3
    simulator._generate_drivers(0.5)
    assert len(simulator.drivers) == 3
    assert all("spawn_node" in driver for driver in simulator.drivers)
    assert all(
        driver["spawn_node"] in simulator.map_loader.road_network
        for driver in simulator.drivers
    )


def test_spot_occupancy_uses_the_simulation_clock(simulator):
    zone = simulator.parking_zones[0]
    assert zone.occupy_spot("driver", current_time=simulator.current_time)
    spot = next(spot for spot in zone.spots if spot.is_occupied)
    assert spot.occupied_since == simulator.current_time


def test_submicrosecond_time_step_is_rejected_instead_of_looping(simulator):
    with pytest.raises(ValueError, match="microsecond"):
        simulator.run_simulation(1, 1e-12)
