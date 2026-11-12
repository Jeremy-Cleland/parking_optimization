"""Capacity-scaled demand experiments with a discarded warm-up and saved charts."""

import json
import time
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from simulation.experiments import ExperimentConfig, run_experiment

LEVELS = {"low": 0.35, "medium": 0.75, "high": 1.15}


def study_figure(reports: dict) -> Figure:
    figure = Figure(figsize=(13, 9), facecolor="#111113", layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).flat
    panels = [
        ("revenue", "Measured revenue (USD)", 1),
        ("mean_occupancy", "Mean occupied capacity (%)", 100),
        ("peak_occupancy", "Peak interval occupancy (%)", 100),
        ("success_rate", "Measurement arrivals who parked (%)", 100),
    ]
    colors = ["#60a5fa", "#34d399", "#c4b5fd", "#fbbf24"]
    for axis, (metric, title, scale) in zip(axes, panels, strict=True):
        axis.set_facecolor("#111113")
        for i, policy in enumerate(next(iter(reports.values()))["config"]["policies"]):
            means, errors = [], [[], []]
            for report in reports.values():
                estimate = report["summary"][policy]["metrics"][metric]
                means.append(estimate["mean"] * scale)
                errors[0].append((estimate["mean"] - estimate["ci95"][0]) * scale)
                errors[1].append((estimate["ci95"][1] - estimate["mean"]) * scale)
            axis.errorbar(
                range(len(reports)),
                means,
                yerr=errors,
                color=colors[i],
                marker="o",
                capsize=4,
                label=policy.replace("-", " + "),
            )
        axis.set_xticks(range(len(reports)), list(reports))
        axis.set_title(title, loc="left", color="#f8fafc", pad=12)
        axis.tick_params(colors="#cbd5e1")
        axis.grid(axis="y", color="#334155", alpha=0.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#334155")
    first = next(iter(reports.values()))
    figure.suptitle(
        f"Simulated parking demand study | {len(first['config']['seeds'])} paired seeds\n{first['config']['duration_hours']:g} hours total, first {first['config']['warmup_hours']:g} hours excluded; means and 95% seed intervals",
        color="#f8fafc",
        fontsize=15,
    )
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=2,
        labelcolor="#f8fafc",
        facecolor="#111113",
        edgecolor="#334155",
    )
    return figure


def run_demand_study(
    output_dir: Path,
    data_source: str = "map",
    data_directory: str = "output/map_data",
    n_zones: int = 8,
    seeds: tuple = (11, 12, 13, 14, 15),
    duration_hours: float = 8,
    warmup_hours: float = 2,
    time_step_minutes: float = 5,
    array_backend: str = "auto",
    workers: int = 4,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reports = {}
    for level, load in LEVELS.items():
        config = ExperimentConfig(
            seeds=seeds,
            n_zones=n_zones,
            n_drivers=0,
            duration_hours=duration_hours,
            warmup_hours=warmup_hours,
            time_step_minutes=time_step_minutes,
            data_source=data_source,
            data_directory=data_directory,
            array_backend=array_backend,
            scenario_event="steady",
            offered_load=load,
        )
        folder = output_dir / ("comparison" if level == "high" else level)
        print(f"Starting {level} demand: offered load {load:g}", flush=True)
        reports[level] = run_experiment(config, folder, workers=workers)
    summary = {
        "scope": "simulation_only",
        "wall_seconds": time.perf_counter() - started,
        "demand_definition": "Offered load = requested arrival rate times expected stay / modeled capacity; it is not an achieved occupancy target.",
        "mean_stay_minutes_for_sizing": 190.75,
        "levels": {
            level: {
                "offered_load": LEVELS[level],
                "experiment_id": report["experiment_id"],
                "config": report["config"],
                "summary": report["summary"],
                "drivers_per_seed": {
                    str(run["seed"]): run["outcomes"]["generated_drivers"]
                    for run in report["runs"]
                },
                "capacity_per_seed": {
                    str(run["seed"]): run["outcomes"]["capacity"]
                    for run in report["runs"]
                },
            }
            for level, report in reports.items()
        },
    }
    (output_dir / "demand_study.json").write_text(json.dumps(summary, indent=2))
    figure = study_figure(reports)
    figure.savefig(
        output_dir / "demand_study.png", dpi=160, facecolor=figure.get_facecolor()
    )
    figure.savefig(
        output_dir / "comparison" / "demand_study.png",
        dpi=160,
        facecolor=figure.get_facecolor(),
    )
    lines = [
        "# Capacity-scaled parking demand study",
        "",
        "Simulation only. Demand and behavior remain uncalibrated to field observations.",
        "",
        summary["demand_definition"],
        "",
        f"Each scenario runs for {duration_hours:g} hours with {warmup_hours:g} hours discarded as warm-up. Pricing and traffic evolve throughout. Measured revenue includes all occupied spaces after warm-up; driver outcomes describe arrivals after warm-up.",
        "",
        "![Demand study](demand_study.png)",
        "",
        "| Demand | Offered load | Drivers per seed | Policy | Mean revenue (USD) | Mean occupancy | Peak occupancy | Parked / measured arrivals |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for level, report in reports.items():
        for policy, estimates in report["summary"].items():
            m = estimates["metrics"]
            count = next(
                r["outcomes"]["generated_drivers"]
                for r in report["runs"]
                if r["policy"] == policy
            )
            lines.append(
                f"| {level} | {LEVELS[level]:.2f} | {count:,} | {policy} | {m['revenue']['mean']:,.2f} | {m['mean_occupancy']['mean']:.1%} | {m['peak_occupancy']['mean']:.1%} | {m['success_rate']['mean']:.1%} |"
            )
    lines += [
        "",
        "Full confidence intervals, paired policy differences, hardware/backend metadata, seed outcomes, raw results and code/data hashes are saved in each scenario's experiment report. Low and medium scenarios are in `low/` and `medium/`; the high scenario is in `comparison/` so the local dashboard can reopen it.",
        "",
        "Interpretation limits: offered demand can exceed available capacity without filling every zone because destinations, walking constraints, accessibility metadata and price choices restrict feasible alternatives. Two hours of warm-up do not prove steady state, particularly for long-stay commuters. Performance claims apply only to measured workloads.",
        "",
    ]
    (output_dir / "demand_study.md").write_text("\n".join(lines))
    return summary
