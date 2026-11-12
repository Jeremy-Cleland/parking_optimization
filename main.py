"""Command-line entry point for the parking optimization research simulator."""

import argparse
import hashlib
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from core.config import get_config
from core.logger import main_logger, metrics
from core.run_manager import complete_run, get_run_path, start_run
from simulation.city_simulator import CitySimulator


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def iso_datetime(value):
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an ISO timestamp, e.g. 2026-11-12T08:00:00"
        ) from exc


def build_parser():
    parser = argparse.ArgumentParser(description="Parking policy research simulator")
    parser.add_argument(
        "--mode",
        choices=[
            "simulate",
            "analyze",
            "visualize",
            "demo",
            "compare",
            "dashboard",
            "study",
        ],
        default="demo",
    )
    parser.add_argument(
        "--zones",
        type=positive_int,
        help="Number of synthetic zones; map mode uses the loaded dataset",
    )
    parser.add_argument(
        "--drivers", type=nonnegative_int, help="Number of simulated drivers"
    )
    parser.add_argument(
        "--duration", type=positive_float, help="Simulation duration in hours"
    )
    parser.add_argument(
        "--time-step", type=positive_float, help="Simulation step in minutes"
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        help="Seed for synthetic scenarios and driver behavior",
    )
    parser.add_argument(
        "--start-time", type=iso_datetime, help="Simulation start as an ISO timestamp"
    )
    parser.add_argument(
        "--data-source",
        choices=["auto", "synthetic", "map"],
        default="auto",
        help="auto falls back to synthetic data; demo defaults to synthetic",
    )
    parser.add_argument("--data-directory", default="output/map_data")
    parser.add_argument(
        "--port", type=positive_int, default=8765, help="Local dashboard port"
    )
    parser.add_argument(
        "--output", help="Also save simulation results to this JSON file"
    )
    parser.add_argument(
        "--input",
        default="output/latest/simulation_results.json",
        help="Existing results JSON for visualize mode",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument(
        "--pricing-policy", choices=["static", "dynamic"], default="dynamic"
    )
    parser.add_argument(
        "--routing-policy", choices=["distance", "time"], default="time"
    )
    parser.add_argument(
        "--traffic-source",
        choices=["simulated", "free-flow", "live"],
        default="simulated",
    )
    parser.add_argument(
        "--event",
        choices=["normal", "event", "steady"],
        default="normal",
        help="Event arrivals cluster in the first quarter of the scenario",
    )
    parser.add_argument("--demand-multiplier", type=positive_float, default=1.0)
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.0,
        help="Initial hours excluded from experiment reporting",
    )
    parser.add_argument(
        "--array-backend", choices=["auto", "numpy", "mlx"], default="auto"
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Independent experiment worker processes",
    )
    parser.add_argument("--min-price", type=positive_float, default=1.0)
    parser.add_argument("--max-price", type=positive_float, default=10.0)
    parser.add_argument("--target-occupancy", type=positive_float, default=0.85)
    parser.add_argument("--max-price-change", type=positive_float, default=0.15)
    parser.add_argument(
        "--seeds",
        default="1,2,3,4,5",
        help="Distinct comma-separated seeds for compare mode",
    )
    parser.add_argument(
        "--policies",
        default="static-distance,dynamic-distance,static-time,dynamic-time",
        help="Comma-separated policies for compare mode; the first is the baseline",
    )
    return parser


def generate_analysis(output_dir):
    from analysis.complexity_analysis import ComplexityAnalyzer

    analyzer = ComplexityAnalyzer(output_dir=str(output_dir))
    analyzer.analyze_all_algorithms()
    analyzer.generate_report()


def generate_visualizations(results_path, output_dir, include_maps=True):
    from analysis.visualizer import ParkingVisualizer

    if not Path(results_path).is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    visualizer = ParkingVisualizer(str(results_path), output_dir=str(output_dir))
    visualizer.create_all_visualizations()
    if include_maps:
        from analysis.map_visualizer import create_all_map_visualizations

        generated = create_all_map_visualizations(
            str(results_path), output_dir=str(output_dir)
        )
        if len(generated) != 3:
            raise RuntimeError(
                f"Only {len(generated)} of 3 map artifacts were generated"
            )


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = get_config()
    demo = args.mode == "demo"
    args.zones = (
        args.zones
        if args.zones is not None
        else (8 if demo else config.simulation.default_zones)
    )
    args.drivers = (
        args.drivers
        if args.drivers is not None
        else (30 if demo else config.simulation.default_drivers)
    )
    args.duration = (
        args.duration
        if args.duration is not None
        else (0.5 if demo else config.simulation.default_duration_hours)
    )
    args.time_step = args.time_step or config.simulation.default_time_step_minutes
    args.seed = (
        args.seed if args.seed is not None else config.simulation.simulation_seed
    )
    start = time.monotonic()
    run_started = False
    sim = None
    try:
        parameters = vars(args).copy()
        if args.start_time is not None:
            parameters["start_time"] = args.start_time.isoformat()
        run_id = start_run(args.mode, parameters)
        run_started = True
        main_logger.info("Starting parking simulation", mode=args.mode, seed=args.seed)
        print(f"Parking optimization | mode={args.mode} | seed={args.seed}")
        errors = []

        if args.mode == "dashboard":
            from simulation.dashboard import serve_dashboard

            serve_dashboard(get_run_path() / "experiments", args.port)
        elif args.mode == "study":
            from simulation.demand_study import run_demand_study

            run_demand_study(
                output_dir=get_run_path(),
                data_source="map" if args.data_source == "map" else "synthetic",
                data_directory=args.data_directory,
                n_zones=args.zones,
                seeds=tuple(int(value.strip()) for value in args.seeds.split(",")),
                duration_hours=args.duration,
                warmup_hours=args.warmup,
                time_step_minutes=args.time_step,
                array_backend=args.array_backend,
                workers=args.workers,
            )
        elif args.mode == "compare":
            from simulation.experiments import ExperimentConfig, run_experiment

            if args.traffic_source != "simulated":
                raise ValueError(
                    "Controlled comparisons use replayable simulated traffic"
                )
            experiment = ExperimentConfig(
                seeds=tuple(int(value.strip()) for value in args.seeds.split(",")),
                policies=tuple(value.strip() for value in args.policies.split(",")),
                n_zones=args.zones,
                n_drivers=args.drivers,
                duration_hours=args.duration,
                time_step_minutes=args.time_step,
                start_time=args.start_time or datetime(2026, 11, 12, 8),
                data_source="map" if args.data_source == "map" else "synthetic",
                data_directory=args.data_directory,
                scenario_event=args.event,
                demand_multiplier=args.demand_multiplier,
                min_price=args.min_price,
                max_price=args.max_price,
                target_occupancy=args.target_occupancy,
                max_price_change=args.max_price_change,
                warmup_hours=args.warmup,
                array_backend=args.array_backend,
            )
            run_experiment(
                experiment, get_run_path() / "comparison", workers=args.workers
            )
            if args.output:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                source = get_run_path() / "comparison" / "experiment.json"
                if source.resolve() != output.resolve():
                    shutil.copy2(source, output)
        elif args.mode in {"simulate", "demo"}:
            use_maps = args.data_source == "map" or (
                args.data_source == "auto" and not demo
            )
            sim = CitySimulator(
                data_directory=args.data_directory,
                n_drivers=args.drivers,
                use_real_data=use_maps,
                n_zones=args.zones,
                seed=args.seed,
                start_time=args.start_time,
                pricing_policy=args.pricing_policy,
                routing_policy=args.routing_policy,
                traffic_source=args.traffic_source,
                scenario_event=args.event,
                demand_multiplier=args.demand_multiplier,
                min_price=args.min_price,
                max_price=args.max_price,
                target_occupancy=args.target_occupancy,
                max_price_change=args.max_price_change,
                warmup_hours=args.warmup,
                array_backend=args.array_backend,
            )
            if args.data_source == "map" and not sim.use_real_data:
                raise ValueError(
                    "Map data could not be loaded; use --data-source synthetic or auto"
                )
            sim.run_simulation(
                duration_hours=args.duration, time_step_minutes=args.time_step
            )
            results_path = get_run_path() / "simulation_results.json"
            sim.save_results(str(results_path))
            if args.output:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.resolve() != results_path.resolve():
                    shutil.copy2(results_path, output)

            if not demo and not args.skip_analysis:
                try:
                    generate_analysis(get_run_path() / "analysis")
                except Exception as exc:
                    errors.append(f"Analysis failed: {exc}")
            if not demo and not args.skip_visualizations:
                try:
                    generate_visualizations(
                        results_path,
                        get_run_path() / "visualizations",
                        include_maps=sim.use_real_data,
                    )
                except Exception as exc:
                    errors.append(f"Visualization failed: {exc}")
        elif args.mode == "analyze":
            generate_analysis(get_run_path() / "analysis")
        elif args.mode == "visualize":
            import json

            source = Path(args.input).resolve()
            with source.open() as stream:
                results = json.load(stream)
            archived_source = get_run_path() / "simulation_results.json"
            if source != archived_source.resolve():
                shutil.copy2(source, archived_source)
            (get_run_path() / "source_provenance.json").write_text(
                json.dumps(
                    {
                        "source_path": str(source),
                        "sha256": hashlib.sha256(
                            archived_source.read_bytes()
                        ).hexdigest(),
                    },
                    indent=2,
                )
            )
            generate_visualizations(
                archived_source,
                get_run_path() / "visualizations",
                include_maps=results.get("simulation_config", {}).get(
                    "use_real_data", False
                ),
            )

        status = "partial" if errors else "completed"
        complete_run(status, time.monotonic() - start)
        run_started = False
        for error in errors:
            main_logger.error(error)
            print(error, file=sys.stderr)
        metrics.log_summary()
        print(f"Run {status}: {run_id}")
        print(f"Artifacts: output/runs/{run_id}")
        return 1 if errors else 0
    except KeyboardInterrupt:
        if run_started:
            complete_run("interrupted", time.monotonic() - start)
        return 130
    except Exception as exc:
        if run_started:
            complete_run("failed", time.monotonic() - start)
        main_logger.error(f"Execution failed: {exc}")
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if sim is not None:
            sim.coordinator.executor.shutdown(wait=True)


if __name__ == "__main__":
    sys.exit(main())
