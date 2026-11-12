"""Paired-seed policy experiments with inspectable observations and uncertainty."""

import csv
import hashlib
import io
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import t

from simulation.city_simulator import CitySimulator

POLICIES = {
    "static-distance": ("static", "distance"),
    "dynamic-distance": ("dynamic", "distance"),
    "static-time": ("static", "time"),
    "dynamic-time": ("dynamic", "time"),
}
METRICS = {
    "revenue": "Simulated occupied-space revenue (USD)",
    "mean_occupancy": "Time-weighted occupied capacity (fraction)",
    "peak_occupancy": "Peak interval occupancy in measurement window (fraction)",
    "success_rate": "Parked / drivers arriving in measurement window (fraction)",
    "rejected": "Rejected drivers (count)",
    "unresolved": "Unresolved drivers (count)",
    "search_minutes": "Search delay, successful drivers only (minutes)",
    "driving_minutes": "Recommended driving time, successful drivers (minutes)",
    "walking_meters": "Estimated walk, successful drivers (meters)",
    "mean_stay_cost": "Quoted full-stay cost, successful drivers (USD)",
    "mean_hourly_price": "Time- and capacity-weighted posted price (USD/hour)",
    "target_occupancy_error": "Time-weighted absolute gap from occupancy target (fraction)",
}


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    policies: tuple[str, ...] = tuple(POLICIES)
    n_zones: int = 8
    n_drivers: int = 100
    duration_hours: float = 2.0
    time_step_minutes: float = 5.0
    start_time: datetime = datetime(2026, 11, 12, 8)
    data_source: str = "synthetic"
    data_directory: str = "output/map_data"
    scenario_event: str = "normal"
    demand_multiplier: float = 1.0
    min_price: float = 1.0
    max_price: float = 10.0
    target_occupancy: float = 0.85
    max_price_change: float = 0.15
    warmup_hours: float = 0.0
    array_backend: str = "auto"
    offered_load: Optional[float] = None

    def __post_init__(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Comparisons require at least two distinct seeds")
        if any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("Seeds must be nonnegative integers")
        if (
            len(self.policies) < 2
            or len(set(self.policies)) != len(self.policies)
            or not set(self.policies) <= POLICIES.keys()
        ):
            raise ValueError("Select at least two distinct known policies")
        if self.n_zones < 1 or self.n_drivers < 0:
            raise ValueError("Zones must be positive and drivers nonnegative")
        for value in (
            self.duration_hours,
            self.time_step_minutes,
            self.demand_multiplier,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Duration, step and demand multiplier must be finite and positive"
                )
        if not 0 < self.min_price < self.max_price or not math.isfinite(self.max_price):
            raise ValueError("Use a finite positive price range")
        if not 0 < self.target_occupancy < 1 or not 0 < self.max_price_change <= 1:
            raise ValueError("Invalid occupancy target or price movement limit")
        if (
            not math.isfinite(self.warmup_hours)
            or not 0 <= self.warmup_hours < self.duration_hours
        ):
            raise ValueError(
                "Warm-up must be finite, nonnegative and shorter than duration"
            )
        if not math.isclose(
            self.warmup_hours * 60 / self.time_step_minutes,
            round(self.warmup_hours * 60 / self.time_step_minutes),
            abs_tol=1e-8,
        ):
            raise ValueError("Warm-up must align with a simulation step boundary")
        if self.offered_load is not None and (
            not math.isfinite(self.offered_load) or self.offered_load <= 0
        ):
            raise ValueError("Offered load must be finite and positive")
        if self.array_backend not in {
            "auto",
            "numpy",
            "mlx",
        } or self.scenario_event not in {"normal", "event", "steady"}:
            raise ValueError("Invalid array backend or arrival pattern")
        if self.data_source not in {"synthetic", "map"}:
            raise ValueError(
                "Experiments require an explicit synthetic or map data source"
            )


def interval(values: List[float]) -> Dict:
    """Two-sided Student t interval across independent scenario seeds."""
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "ci95": None}
    margin = float(
        t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values))
    )
    return {"n": len(values), "mean": mean, "ci95": [mean - margin, mean + margin]}


def summarize(results: Dict) -> Dict:
    metrics = results["metrics"]
    config = results["simulation_config"]
    outcomes = results["driver_outcomes"]
    ends = np.asarray(metrics["elapsed_minutes"], dtype=float)
    starts = np.concatenate(([0], ends[:-1]))
    durations = ends - starts
    cutoff = config.get("warmup_hours", 0) * 60
    weights = np.maximum(0, ends - np.maximum(starts, cutoff))
    ratios = weights / durations

    def weighted(values: List[float]) -> Optional[float]:
        return (
            float(np.average(values, weights=weights))
            if len(values) and weights.sum()
            else None
        )

    def mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    records = results.get("driver_records")
    observations = metrics.get("parking_observations")
    if cutoff and (records is None or observations is None):
        raise ValueError("Warm-up reporting requires driver and parking observations")
    if records is not None:
        cohort = [row for row in records if row["arrival_minutes"] >= cutoff]
        cohort_ids = {row["id"] for row in cohort}
        parked = sum(row["parked_at_minutes"] is not None for row in cohort)
        rejected = sum(
            row["parked_at_minutes"] is None and row["status"] == "departed"
            for row in cohort
        )
        outcomes = {
            "total": len(cohort),
            "parked": parked,
            "rejected": rejected,
            "unresolved": sum(
                row["status"] in {"searching", "traveling"} for row in cohort
            ),
        }
        observations = [
            row for row in observations or [] if row["driver_id"] in cohort_ids
        ]
    observed = {
        "search_minutes": mean([row["search_minutes"] for row in observations])
        if observations is not None
        else mean(metrics["avg_search_time"]),
        "search_n": len(observations)
        if observations is not None
        else len(metrics["avg_search_time"]),
        **{
            name: mean([row[name] for row in observations])
            if observations is not None
            else mean(metrics[source])
            for name, source in [
                ("driving_minutes", "driving_minutes"),
                ("walking_meters", "walking_meters"),
                ("mean_stay_cost", "estimated_parking_cost"),
            ]
        },
    }
    capacities = {
        z["id"]: z["capacity"] for z in results["final_state"]["zone_occupancy"]
    }
    total_capacity = sum(capacities.values())
    posted_prices = [
        sum(prices[z] * c for z, c in capacities.items()) / total_capacity
        for prices in metrics["zone_prices"]
    ]
    return {
        "revenue": float(np.dot(metrics["total_revenue"], ratios)),
        "mean_occupancy": weighted(metrics["avg_occupancy"]),
        "peak_occupancy": max(
            (
                value
                for value, weight in zip(metrics["avg_occupancy"], weights, strict=True)
                if weight > 0
            ),
            default=None,
        ),
        "capacity": total_capacity,
        "generated_drivers": results["driver_outcomes"]["total"],
        "actual_offered_load": sum(row["parking_duration_minutes"] for row in records)
        / (total_capacity * config["duration_hours"] * 60)
        if records and "parking_duration_minutes" in records[0]
        else None,
        "measurement_hours": float(weights.sum() / 60),
        "success_rate": outcomes["parked"] / outcomes["total"]
        if outcomes["total"]
        else None,
        "rejected": outcomes["rejected"],
        "unresolved": outcomes["unresolved"],
        "mean_hourly_price": weighted(posted_prices),
        "target_occupancy_error": weighted(
            [
                abs(x - config["price_constraints"]["target_occupancy"])
                for x in metrics["avg_occupancy"]
            ]
        ),
        **observed,
        **outcomes,
    }


def aggregate(runs: List[Dict], policies: tuple) -> Dict:
    """Aggregate seed outcomes and pair each policy against the first policy."""
    baseline = {r["seed"]: r for r in runs if r["policy"] == policies[0]}
    summaries = {}
    for policy in policies:
        selected = [r for r in runs if r["policy"] == policy]
        metrics = {}
        differences = {}
        for metric in METRICS:
            metrics[metric] = interval(
                [
                    r["outcomes"][metric]
                    for r in selected
                    if r["outcomes"][metric] is not None
                ]
            )
            differences[metric] = interval(
                [
                    r["outcomes"][metric] - baseline[r["seed"]]["outcomes"][metric]
                    for r in selected
                    if r["outcomes"][metric] is not None
                    and baseline[r["seed"]]["outcomes"][metric] is not None
                ]
            )
        summaries[policy] = {
            "metrics": metrics,
            "paired_difference_from_baseline": differences,
        }
    return summaries


def _execute_run(job: tuple) -> Dict:
    config, output_dir, seed, policy = job
    pricing, routing = POLICIES[policy]
    sim = None
    prefix = f"{policy}/seed-{seed}"
    artifact = output_dir / prefix / "results.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (artifact.parent / "simulation.log").open("w") as log, redirect_stdout(log):
        try:
            sim = CitySimulator(
                n_zones=config.n_zones,
                n_drivers=config.n_drivers,
                seed=seed,
                start_time=config.start_time,
                use_real_data=config.data_source == "map",
                data_directory=config.data_directory,
                pricing_policy=pricing,
                routing_policy=routing,
                traffic_source="simulated",
                scenario_event=config.scenario_event,
                demand_multiplier=config.demand_multiplier,
                min_price=config.min_price,
                max_price=config.max_price,
                target_occupancy=config.target_occupancy,
                max_price_change=config.max_price_change,
                warmup_hours=config.warmup_hours,
                array_backend=config.array_backend,
            )
            if config.data_source == "map" and not sim.use_real_data:
                raise ValueError(
                    "Comparison requires valid map data; synthetic fallback is disabled"
                )
            if config.offered_load is not None:
                capacity = sum(zone.capacity for zone in sim.parking_zones)
                # Mean stay from the current commuter/errand/tourist/student mix.
                sim.n_drivers = math.ceil(
                    config.offered_load * capacity * config.duration_hours * 60 / 190.75
                )
            sim.run_simulation(config.duration_hours, config.time_step_minutes)
            results = sim.results()
            sim.save_results(str(artifact))
        finally:
            if sim is not None:
                sim.coordinator.executor.shutdown(wait=True)
    (artifact.parent / "performance.json").write_text(
        json.dumps({"wall_seconds": time.perf_counter() - started}, indent=2)
    )
    return {
        "seed": seed,
        "policy": policy,
        "scenario_fingerprint": results["scenario_fingerprint"],
        "artifact": f"{prefix}/results.json",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "outcomes": summarize(results),
        "compute": results["compute"],
    }


def run_experiment(
    config: ExperimentConfig, output_dir: Path, workers: int = 1
) -> Dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (config, output_dir, seed, policy)
        for seed in config.seeds
        for policy in config.policies
    ]
    runs = []
    signatures: dict[int, str] = {}
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        completed = pool.map(_execute_run, jobs) if pool else map(_execute_run, jobs)
        for run in completed:
            seed, policy = run["seed"], run["policy"]
            if seed in signatures and signatures[seed] != run["scenario_fingerprint"]:
                raise RuntimeError("Policies did not receive identical scenario inputs")
            signatures[seed] = run["scenario_fingerprint"]
            runs.append(run)
            print(f"Completed {policy}, seed {seed}", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    results = json.loads((output_dir / runs[0]["artifact"]).read_text())
    settings = asdict(config)
    settings["start_time"] = config.start_time.isoformat()
    source_root = Path(__file__).resolve().parents[1]
    source_paths = sorted(
        [*source_root.glob("core/*.py"), *source_root.glob("simulation/*.py")]
    )
    report = {
        "schema_version": 1,
        "scope": "simulation_only",
        "config": settings,
        "baseline": config.policies[0],
        "metric_definitions": METRICS,
        "uncertainty": "95% Student t intervals across seeds; paired differences use identical seeds. These intervals do not capture model uncertainty. Small seed counts are exploratory.",
        "software": {
            name: version(name)
            for name in ("parking-optimization", "numpy", "networkx", "scipy")
        },
        "source_sha256": {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "assumptions": results["assumptions"],
        "runs": runs,
        "summary": aggregate(runs, config.policies),
    }
    report["experiment_id"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()
    ).hexdigest()[:16]
    (output_dir / "experiment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    write_report(report, output_dir)
    return report


def write_report(report: Dict, output_dir: Path) -> None:
    """Export a decision report and seed-level CSV beside the raw observations."""
    from simulation.experiment_charts import save_comparison_chart

    save_comparison_chart(report, output_dir / "policy_comparison.png")
    lines = [
        "# Simulated parking policy comparison",
        "",
        f"Experiment `{report['experiment_id']}`. Baseline: `{report['baseline']}`.",
        "",
        report["uncertainty"],
        "",
        "![Policy means and 95% confidence intervals](policy_comparison.png)",
        "",
        "Warm-up revenue and occupancy are excluded. Driver outcomes and conditional means describe arrivals during the measurement window; unresolved late arrivals remain in the denominator.",
        "",
        "## Scenario",
        "",
        "```json",
        json.dumps(report["config"], indent=2),
        "```",
        "",
        "## Outcomes and trade-offs",
        "",
    ]
    for metric, label in report.get("metric_definitions", METRICS).items():
        lines.extend(
            [
                f"### {label}",
                "",
                "| Policy | Mean (95% CI) | Seeds | Paired difference vs baseline (95% CI) |",
                "|---|---:|---:|---:|",
            ]
        )
        for policy, summary in report["summary"].items():
            estimate = summary["metrics"][metric]
            delta = summary["paired_difference_from_baseline"][metric]

            def fmt(item: Dict) -> str:
                if item["mean"] is None:
                    return "No observations"
                ci = item["ci95"]
                return f"{item['mean']:.4f}" + (
                    f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else " (CI unavailable)"
                )

            lines.append(
                f"| {policy} | {fmt(estimate)} | {estimate['n']} | {fmt(delta)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Compare revenue, availability and cost together; no single objective establishes the best policy. A paired interval containing zero is inconclusive for that metric. Search, drive, walk and cost means describe successful drivers only; each policy may serve a different driver population. Seed-level CSV includes all driver outcome counts and successful-search sample sizes.",
            "",
            "## Modeling limits",
            "",
        ]
    )
    lines.extend(f"- {assumption}" for assumption in report["assumptions"])
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "`experiment.json` records policy settings, source-code hashes, software versions, per-seed outcomes, confidence intervals and SHA-256 hashes of each raw results file. Each policy/seed directory contains `results.json` and `simulation.log`. Forecast evaluation is a synthetic chronological holdout, not evidence of field accuracy.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))
    stream = io.StringIO(newline="")
    fields = ["policy", "seed", *report["runs"][0]["outcomes"]]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for run in report["runs"]:
        writer.writerow(
            {"policy": run["policy"], "seed": run["seed"], **run["outcomes"]}
        )
    (output_dir / "seed_outcomes.csv").write_text(stream.getvalue())
