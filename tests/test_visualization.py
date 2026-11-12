"""Ensure charts reflect the recorded observations and interval units."""

import json

import matplotlib.pyplot as plt
import numpy as np

from analysis.visualizer import ParkingVisualizer


def test_recorded_zone_occupancy_is_rendered_without_synthetic_variation(
    tmp_path, monkeypatch
):
    result = {
        "simulation_config": {"time_step_minutes": 10},
        "metrics": {
            "avg_occupancy": [0.5, 0.5],
            "elapsed_minutes": [10, 15],
            "total_revenue": [1, 0.5],
            "zone_occupancy": [{"a": 0.2, "b": 0.8}, {"a": 0.4, "b": 0.6}],
        },
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    visualizer = ParkingVisualizer(str(path))
    captured = []

    def capture(figure, _path):
        captured.append(figure)

    monkeypatch.setattr("analysis.visualizer.save_plot", capture)
    visualizer.plot_occupancy_heatmap()
    figure = captured[0]
    try:
        np.testing.assert_array_equal(
            figure.axes[0].images[0].get_array(), [[0.2, 0.4], [0.8, 0.6]]
        )
        assert visualizer._elapsed_minutes() == [10, 15]
    finally:
        plt.close(figure)


def test_legacy_results_without_zone_history_do_not_invent_a_heatmap(
    tmp_path, monkeypatch
):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"metrics": {"avg_occupancy": [0.5]}}))
    visualizer = ParkingVisualizer(str(path))

    def unexpected_plot(*args):
        raise AssertionError("No per-zone observations were recorded")

    monkeypatch.setattr("analysis.visualizer.save_plot", unexpected_plot)
    visualizer.plot_occupancy_heatmap()


def test_hourly_revenue_preserves_totals_across_long_and_partial_steps(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "avg_occupancy": [0.5, 0.5],
                    "elapsed_minutes": [75, 90],
                    "total_revenue": [75, 15],
                }
            }
        )
    )
    visualizer = ParkingVisualizer(str(path))
    assert visualizer._hourly_revenue() == [60, 30]


def test_requested_output_directory_is_used_from_construction(tmp_path, monkeypatch):
    import main

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"metrics": {}}))
    monkeypatch.setattr(
        ParkingVisualizer, "create_all_visualizations", lambda self: None
    )
    main.generate_visualizations(path, tmp_path / "charts", include_maps=False)
    assert (tmp_path / "charts").is_dir()
    assert not (tmp_path / "visualization_output").exists()


def test_map_occupancy_uses_recorded_final_state_and_rejects_missing_data():
    import pytest

    from analysis.map_visualizer import MapVisualizer

    visualizer = MapVisualizer.__new__(MapVisualizer)
    visualizer.results = {
        "final_state": {
            "zone_occupancy": [{"id": "lot_1", "capacity": 20, "occupied": 7}]
        }
    }
    assert visualizer._get_zone_occupancy("lot_1") == 0.35
    with pytest.raises(ValueError, match="No recorded occupancy"):
        visualizer._get_zone_occupancy("missing")


def test_map_revenue_and_capacity_charts_use_supported_api_and_observations():
    from types import SimpleNamespace

    from analysis.map_visualizer import MapVisualizer
    from analysis.viz_utils import get_theme_colors

    visualizer = MapVisualizer.__new__(MapVisualizer)
    visualizer.theme_colors = get_theme_colors()
    visualizer.results = {
        "metrics": {"zone_revenue": [{"lot": 1, "meter": 2}, {"lot": 3, "meter": 4}]}
    }
    visualizer.map_loader = SimpleNamespace(
        get_parking_zones=lambda: [
            {"id": "lot", "type": "lot", "capacity": 20},
            {"id": "meter", "type": "meter", "capacity": 8},
        ]
    )
    figure, axes = plt.subplots(1, 2)
    try:
        visualizer._plot_revenue_by_area(axes[0])
        assert [bar.get_height() for bar in axes[0].patches] == [4, 6]
        visualizer._plot_capacity_distribution(axes[1])
        assert [label.get_text() for label in axes[1].get_xticklabels()] == [
            "Lots",
            "Meters",
        ]
    finally:
        plt.close(figure)


def test_map_dataset_must_match_recorded_input_hashes():
    from types import SimpleNamespace

    import pytest

    from analysis.map_visualizer import MapVisualizer

    visualizer = MapVisualizer.__new__(MapVisualizer)
    visualizer.results = {"data_validation": {"files": {"lots.geojson": "old"}}}
    visualizer.map_loader = SimpleNamespace(
        load_all_data=lambda: True, validation_report={"files": {"lots.geojson": "new"}}
    )
    with pytest.raises(ValueError, match="hashes do not match"):
        visualizer._load_map_data()
