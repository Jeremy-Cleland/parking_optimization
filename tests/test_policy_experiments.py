"""Behavioral contracts for controlled experiments and the policies they exercise."""

import hashlib
import json
from datetime import datetime, timedelta

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from core.demand_predictor import DemandPredictor
from core.dynamic_pricing import DynamicPricingEngine
from core.map_data_loader import MapDataLoader
from core.parking_zone import ParkingZone
from core.route_optimizer import RouteOptimizer
from core.traffic_manager import TrafficCondition, TrafficManager
from simulation.city_simulator import CitySimulator
from simulation.experiments import ExperimentConfig, interval, run_experiment, summarize

NOW = datetime(2026, 11, 12, 8)


def test_forecast_scales_with_capacity_without_random_draws():
    predictor = DemandPredictor(capacity=20)
    predictor.dp_table.fill(0.4)
    state = np.random.get_state()
    small = predictor.predict_demand(NOW, 0.5, capacity=10)
    large = predictor.predict_demand(NOW, 0.5, capacity=40)
    assert small == [4] * 4
    assert large == [16] * 4
    assert small == predictor.predict_demand(NOW, 0.5, capacity=10)
    assert np.array_equal(state[1], np.random.get_state()[1])
    with pytest.raises(ValueError):
        predictor.predict_demand(NOW, 0.5, capacity=0)


def test_forecast_training_and_holdout_have_explicit_arrival_units():
    predictor = DemandPredictor(capacity=20)
    row = {"timestamp": NOW, "occupancy_rate": 0.5, "arrivals": 8, "capacity": 20}
    predictor.train_model([row])
    assert predictor.predict_demand(NOW, 0.5, lookahead_hours=1) == [8]
    assert predictor.evaluate([row]) == {"n": 1, "mae_arrivals": 0, "wape": 0}


def test_pricing_is_order_invariant_and_forecasts_influence_it():
    zones = [
        ParkingZone(str(i), str(i), (42.96, -85.66 + i / 10000), 10, price)
        for i, price in enumerate([2, 8, 4])
    ]
    engine = DynamicPricingEngine()
    snapshot = [z.current_price for z in zones]
    forward = engine.optimize_city_pricing(zones, NOW, apply=False)
    reverse = engine.optimize_city_pricing(list(reversed(zones)), NOW, apply=False)
    assert forward == reverse
    assert [z.current_price for z in zones] == snapshot
    low = engine.optimize_city_pricing(
        zones, NOW, {z.id: 0 for z in zones}, apply=False
    )
    high = engine.optimize_city_pricing(
        zones, NOW, {z.id: 20 for z in zones}, apply=False
    )
    assert all(high[z.id] > low[z.id] for z in zones)
    assert engine.bound_price(4, 100) == pytest.approx(4.6)


@pytest.fixture
def router(tmp_path):
    loader = MapDataLoader(str(tmp_path))
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for i in range(3):
        graph.add_node(i, y=42.96 + i * 0.001, x=-85.66)
    graph.add_edge(0, 2, length=150, travel_time=10)
    graph.add_edge(0, 1, length=100, travel_time=6)
    graph.add_edge(1, 2, length=100, travel_time=6)
    loader.road_network = graph
    loader._parking_zones = [
        {
            "id": "z",
            "coordinates": (42.962, -85.66),
            "hourly_rate": 4,
            "accessible": True,
        }
    ]
    return RouteOptimizer(loader)


def test_traffic_changes_path_does_not_compound_and_expires(router):
    assert router.find_optimal_route(
        (42.96, -85.66), preferences={"prefer_time": True}
    ).path_nodes == [0, 2]
    observation = TrafficCondition("edge", 10, 2, False, 3, NOW)
    router.update_traffic_conditions({(0, 2, 0): observation}, NOW)
    router.update_traffic_conditions({(0, 2, 0): observation}, NOW)
    assert router.road_graph[0][2][0]["travel_time"] == 30
    assert router.find_optimal_route(
        (42.96, -85.66), preferences={"prefer_time": True}
    ).path_nodes == [0, 1, 2]
    assert router.road_graph[0][1][0]["travel_time"] == 6
    router.expire_traffic(NOW + timedelta(minutes=6))
    assert router.road_graph[0][2][0]["travel_time"] == 10
    router.update_traffic_conditions(
        {(0, 2, 0): observation}, NOW + timedelta(minutes=6)
    )
    assert router.road_graph[0][2][0]["travel_time"] == 10


def test_destination_cost_and_accessibility_constraints(router):
    preferences = {
        "destination": (42.962, -85.66),
        "max_walk_distance": 0.05,
        "parking_duration_minutes": 120,
        "zone_prices": {"z": 3},
        "requires_accessible": True,
    }
    route = router.find_optimal_parking((42.96, -85.66), preferences)["z"]
    assert route.total_distance == 150
    assert route.walking_distance == 0
    assert route.estimated_parking_cost == 6
    assert (
        router.find_optimal_parking(
            (42.96, -85.66), {**preferences, "destination": (43, -85.66)}
        )
        == {}
    )
    assert (
        router.find_optimal_parking(
            (42.96, -85.66), {**preferences, "max_parking_cost": 5}
        )
        == {}
    )
    router.parking_zones[0]["accessible"] = False
    assert router.find_optimal_parking((42.96, -85.66), preferences) == {}


def test_provider_incident_counts_do_not_masquerade_as_measured_delays():
    manager = TrafficManager(np_rng=np.random.default_rng(1))
    manager.map_provider_name = "google"
    assert manager._adapt_service_output({"incidents": [{}, {}]}, "edge") is None
    condition = manager._adapt_service_output(
        {
            "routes": [
                {
                    "legs": [
                        {
                            "duration": {"value": 100},
                            "duration_in_traffic": {"value": 150},
                        }
                    ]
                }
            ]
        },
        "edge",
    )
    assert condition.is_real_data
    assert condition.travel_time_factor == 1.5
    assert "proxy" in condition.source


def test_geographic_validation_metric_area_units_and_missing_values(tmp_path):
    loader = MapDataLoader(str(tmp_path))
    polygon = box(-85.67, 42.96, -85.669, 42.961)
    row = pd.Series({"geometry": polygon, "capacity": np.nan, "hourly_rate": 0.5})
    capacity, source = loader._validated_capacity(row, "EPSG:4326", False)
    assert capacity > 100  # Degree-squared would have yielded the old 10-space default.
    assert source == "estimate:metric_area_m2/25"
    assert loader._validated_rate(row, 2)[0] == 0.5
    assert (
        loader._validated_rate(pd.Series({"rate": 0.5, "rate_unit": "minute"}), 2)[0]
        == 30
    )
    assert (
        loader._validated_capacity(
            pd.Series({"geometry": Point(-85.66, 42.96), "spots": 20}),
            "EPSG:4326",
            False,
        )[0]
        == 20
    )
    for bad in [-1, float("inf"), "bad"]:
        with pytest.raises(ValueError):
            loader._validated_rate(pd.Series({"hourly_rate": bad}), 2)
    with pytest.raises(ValueError):
        loader._normalize_geometry(gpd.GeoDataFrame(geometry=[polygon]), "test")
    projected = gpd.GeoDataFrame(geometry=[polygon], crs=4326).to_crs(3857)
    normalized = loader._normalize_geometry(projected, "test")
    assert normalized.crs.to_epsg() == 4326


def test_intervals_do_not_invent_uncertainty_for_missing_or_one_seed():
    assert interval([]) == {"n": 0, "mean": None, "ci95": None}
    assert interval([2])["ci95"] is None
    summary = interval([1, 2, 3, 4, 5])
    assert summary["mean"] == 3
    assert summary["ci95"] == pytest.approx([1.0367568385, 4.9632431615])


def test_paired_experiments_replay_inputs_and_archive_observations(tmp_path):
    config = ExperimentConfig(
        seeds=(7, 8),
        policies=("static-distance", "dynamic-time"),
        n_zones=3,
        n_drivers=12,
        duration_hours=0.3,
        time_step_minutes=7,
    )
    report = run_experiment(config, tmp_path / "first")
    repeated = run_experiment(config, tmp_path / "second")
    assert report == repeated
    for seed in config.seeds:
        runs = [run for run in report["runs"] if run["seed"] == seed]
        assert len({run["scenario_fingerprint"] for run in runs}) == 1
    for run in report["runs"]:
        artifact = tmp_path / "first" / run["artifact"]
        assert run["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        outcomes = run["outcomes"]
        assert (
            outcomes["total"]
            == outcomes["parked"] + outcomes["rejected"] + outcomes["unresolved"]
        )
    baseline = report["summary"]["static-distance"]["paired_difference_from_baseline"]
    assert baseline["revenue"]["mean"] == 0
    assert baseline["revenue"]["ci95"] == [0, 0]
    assert (tmp_path / "first" / "report.md").exists()
    assert (
        (tmp_path / "first" / "policy_comparison.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )
    assert (tmp_path / "first" / "seed_outcomes.csv").exists()
    assert (
        json.loads((tmp_path / "first" / "experiment.json").read_text())["scope"]
        == "simulation_only"
    )


def test_time_weighted_occupancy_and_total_driver_denominators():
    results = {
        "simulation_config": {"price_constraints": {"target_occupancy": 0.8}},
        "driver_outcomes": {"total": 10, "parked": 2, "rejected": 3, "unresolved": 5},
        "final_state": {"zone_occupancy": [{"id": "z", "capacity": 10}]},
        "metrics": {
            "elapsed_minutes": [10, 15],
            "avg_occupancy": [0.2, 0.8],
            "total_revenue": [2, 1],
            "zone_prices": [{"z": 2}, {"z": 8}],
            "avg_search_time": [],
            "driving_minutes": [],
            "walking_meters": [],
            "estimated_parking_cost": [],
        },
    }
    outcomes = summarize(results)
    assert outcomes["mean_occupancy"] == pytest.approx(0.4)
    assert outcomes["success_rate"] == 0.2
    assert outcomes["search_minutes"] is None
    assert outcomes["revenue"] == 3


def test_partial_interval_revenue_stops_at_departure():
    sim = CitySimulator(n_zones=1, n_drivers=0, use_real_data=False, seed=1)
    try:
        zone = sim.parking_zones[0]
        zone.current_price = 6
        zone.occupy_spot("short", current_time=sim.current_time)
        sim.drivers = [
            {"id": "short", "park_start_time": sim.current_time, "duration_minutes": 2}
        ]
        sim._update_metrics(5)
        assert sim.metrics["total_revenue"] == [0.2]
        assert sim.metrics["avg_occupancy"] == pytest.approx([0.4 / zone.capacity])
        assert sim.metrics["zone_occupancy"][0][zone.id] == pytest.approx(
            0.4 / zone.capacity
        )
    finally:
        sim.coordinator.executor.shutdown(wait=True)


def test_accepted_entry_rate_survives_posted_price_changes():
    sim = CitySimulator(n_zones=1, n_drivers=0, use_real_data=False, seed=1)
    try:
        zone = sim.parking_zones[0]
        zone.occupy_spot("quoted", current_time=sim.current_time)
        sim.drivers = [
            {
                "id": "quoted",
                "park_start_time": sim.current_time,
                "duration_minutes": 30,
                "accepted_hourly_rate": 3,
            }
        ]
        zone.current_price = 12
        sim._update_metrics(5)
        zone.current_price = 1
        sim._update_metrics(5)
        assert sim.metrics["total_revenue"] == [0.25, 0.25]
    finally:
        sim.coordinator.executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": (1, 1)},
        {"seeds": (1,)},
        {"policies": ("unknown", "dynamic-time")},
        {"data_source": "auto"},
        {"max_price": float("inf")},
        {"target_occupancy": 1},
        {"duration_hours": 0},
    ],
)
def test_invalid_experiments_fail_before_running(changes):
    with pytest.raises(ValueError):
        ExperimentConfig(**changes)


def test_dashboard_comparison_exports_and_rejects_external_requests(tmp_path):
    import threading
    from http.server import HTTPServer

    import requests

    from simulation.dashboard import make_handler

    server = HTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    payload = {"seeds": [2, 3], "n_zones": 2, "n_drivers": 4, "duration_hours": 0.2}
    try:
        page = requests.get(url, timeout=5)
        assert page.status_code == 200
        assert 'name="scenario_event"' in page.text
        assert (
            requests.post(url + "/compare", json=payload, timeout=5).status_code == 403
        )
        response = requests.post(
            url + "/compare", json=payload, headers={"Origin": url}, timeout=30
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["report"]["runs"]) == 8
        assert (
            requests.get(url + "/latest", timeout=5).json()["report"]["experiment_id"]
            == body["report"]["experiment_id"]
        )
        for artifact in [
            "report.md",
            "experiment.json",
            "seed_outcomes.csv",
            "policy_comparison.png",
        ]:
            assert (
                requests.get(
                    url + body["artifact_base"] + "/" + artifact, timeout=5
                ).status_code
                == 200
            )
        assert (
            requests.get(
                url + "/artifacts/%2e%2e/pyproject.toml", timeout=5
            ).status_code
            == 404
        )
        rejected = requests.post(
            url + "/compare",
            json={**payload, "n_zones": 1000},
            headers={"Origin": url},
            timeout=5,
        )
        assert rejected.status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_compare_cli_archives_a_complete_experiment(tmp_path, monkeypatch):
    import main
    from core.run_manager import RunManager

    manager = RunManager(str(tmp_path / "runs"))
    monkeypatch.setattr(main, "start_run", manager.start_run)
    monkeypatch.setattr(main, "get_run_path", manager.get_run_path)
    monkeypatch.setattr(main, "complete_run", manager.complete_run)
    exported = tmp_path / "export.json"
    assert (
        main.main(
            [
                "--mode",
                "compare",
                "--zones",
                "2",
                "--drivers",
                "4",
                "--duration",
                "0.2",
                "--seeds",
                "2,3",
                "--policies",
                "static-distance,dynamic-time",
                "--output",
                str(exported),
            ]
        )
        == 0
    )
    assert len(json.loads(exported.read_text())["runs"]) == 4
    assert manager.list_runs()[0]["status"] == "completed"


def test_event_window_and_demand_scale_change_forecasts():
    predictor = DemandPredictor(capacity=20)
    predictor.dp_table.fill(0.4)
    predictor.event_windows = [(NOW, NOW + timedelta(minutes=30), 3.2)]
    predictor.demand_scale = 2
    assert predictor.predict_demand(NOW, 0.5, lookahead_hours=2) == pytest.approx(
        [51.2, 16]
    )


def test_comparison_chart_plots_recorded_means_and_intervals():
    from simulation.experiment_charts import PANELS, comparison_figure

    report = {
        "config": {"seeds": [1, 2], "policies": ["static-distance", "dynamic-time"]},
        "summary": {},
    }
    for policy, mean in [("static-distance", 2), ("dynamic-time", 3)]:
        report["summary"][policy] = {
            "metrics": {
                metric: {"mean": mean, "ci95": [mean - 0.5, mean + 0.5], "n": 2}
                for metric, _, _, _ in PANELS
            }
        }
    figure = comparison_figure(report)
    try:
        for axis, (_, _, _, scale) in zip(figure.axes, PANELS, strict=True):
            assert [
                float(container.lines[0].get_xdata()[0])
                for container in axis.containers
            ] == [2 * scale, 3 * scale]
            np.testing.assert_allclose(
                axis.containers[0].lines[2][0].get_segments()[0][:, 0],
                [1.5 * scale, 2.5 * scale],
            )
        report["summary"]["static-distance"]["metrics"]["revenue"] = {
            "mean": None,
            "ci95": None,
            "n": 0,
        }
        empty = comparison_figure(report)
        assert "No observations" in [text.get_text() for text in empty.axes[0].texts]
        empty.clear()
    finally:
        figure.clear()


def test_dashboard_restores_previous_comparison_without_rerunning(tmp_path):
    from simulation.dashboard import make_handler, restore_latest_comparison

    source = tmp_path / "runs" / "run_2026-11-12_08-00-00" / "comparison"
    report = run_experiment(
        ExperimentConfig(
            seeds=(1, 2),
            policies=("static-distance", "dynamic-time"),
            n_zones=2,
            n_drivers=2,
            duration_hours=0.1,
        ),
        source,
    )
    destination = tmp_path / "runs" / "run_2026-11-12_09-00-00" / "experiments"
    assert restore_latest_comparison(destination)
    saved = destination / "saved-comparison"
    assert (saved / "experiment.json").read_bytes() == (
        source / "experiment.json"
    ).read_bytes()
    assert (
        json.loads((saved / "experiment.json").read_text())["experiment_id"]
        == report["experiment_id"]
    )
    assert (saved / "policy_comparison.png").exists()
    assert (saved / report["runs"][0]["artifact"]).read_bytes() == (
        source / report["runs"][0]["artifact"]
    ).read_bytes()
    assert make_handler(destination).latest_directory == saved.resolve()
    # The newest result may have been created in the browser rather than the CLI.
    dashboard_source = (
        tmp_path
        / "runs"
        / "run_2026-11-12_09-30-00"
        / "experiments"
        / "experiment-latest"
    )
    dashboard_report = run_experiment(
        ExperimentConfig(
            seeds=(3, 4),
            policies=("static-distance", "dynamic-time"),
            n_zones=2,
            n_drivers=2,
            duration_hours=0.1,
        ),
        dashboard_source,
    )
    next_session = tmp_path / "runs" / "run_2026-11-12_10-00-00" / "experiments"
    assert restore_latest_comparison(next_session)
    restored = next_session / "saved-comparison"
    assert (
        json.loads((restored / "experiment.json").read_text())["experiment_id"]
        == dashboard_report["experiment_id"]
    )
    assert (restored / "experiment.json").read_bytes() == (
        dashboard_source / "experiment.json"
    ).read_bytes()
