"""
Real-Time Collaborative Parking Space Optimization
Main execution script for CIS 505 Term Project
"""

import argparse
import sys
import time
from datetime import datetime

from analysis.complexity_analysis import ComplexityAnalyzer
from analysis.visualizer import ParkingVisualizer
from core.config import get_config
from core.logger import main_logger, metrics
from core.run_manager import (
    complete_run,
    get_run_path,
    save_artifact,
    start_run,
)
from simulation.city_simulator import CitySimulator


def main():
    """Main entry point for parking optimization system"""
    try:
        # Load configuration
        config = get_config()

        parser = argparse.ArgumentParser(
            description="Real-Time Parking Optimization System"
        )

        parser.add_argument(
            "--mode",
            choices=["simulate", "analyze", "visualize", "demo"],
            default="demo",
            help="Execution mode",
        )

        parser.add_argument(
            "--zones",
            type=int,
            default=config.simulation.default_zones,
            help="Number of parking zones",
        )

        parser.add_argument(
            "--drivers",
            type=int,
            default=config.simulation.default_drivers,
            help="Number of simulated drivers",
        )

        parser.add_argument(
            "--duration",
            type=float,
            default=config.simulation.default_duration_hours,
            help="Simulation duration in hours",
        )

        parser.add_argument(
            "--output",
            type=str,
            default=str(config.output.output_dir / "results.json"),
            help="Output file for results",
        )

        args = parser.parse_args()

        # Start tracking this run
        start_time = time.time()
        run_parameters = {
            "zones": args.zones,
            "drivers": args.drivers,
            "duration": args.duration,
            "output": args.output,
        }
        run_id = start_run(args.mode, run_parameters)

        # Log startup
        main_logger.info("Starting parking optimization system")
        main_logger.info(
            "System configuration",
            mode=args.mode,
            zones=args.zones,
            drivers=args.drivers,
            duration_hours=args.duration,
            api_keys_available=config.has_api_keys,
        )

        print("=" * 60)
        print("REAL-TIME COLLABORATIVE PARKING SPACE OPTIMIZATION")
        print("CIS 505 - Algorithms Analysis and Design")
        print("=" * 60)
        print(f"\nStarting at: {datetime.now()}")
        print(f"Mode: {args.mode}")
        print(f"Configuration: {args.zones} zones, {args.drivers} drivers")
        print(
            f"API Status: {'✅ Connected' if config.has_api_keys else '⚠️  Fallback mode'}"
        )

        # Show configuration warnings
        warnings = config.validate()
        for warning in warnings:
            print(f"⚠️  {warning}")
        print()

        if args.mode == "simulate" or args.mode == "demo":
            # Run simulation
            print("Initializing city simulation...")
            sim = CitySimulator(
                data_directory="output/map_data",
                n_drivers=args.drivers,
                use_real_data=True,
            )

            print("Running simulation...")
            sim.run_simulation(duration_hours=args.duration, time_step_minutes=5)

            # Save results to run directory
            results_path = get_run_path() / "simulation_results.json"
            sim.save_results(str(results_path))

            # Always run analysis and visualization after simulation
            print("\nPerforming complexity analysis...")
            try:
                # Create analysis directory in run folder
                analysis_dir = get_run_path() / "analysis"
                analysis_dir.mkdir(exist_ok=True)

                # Run complexity analysis directly to run directory
                analyzer = ComplexityAnalyzer(output_dir=str(analysis_dir))
                analyzer.analyze_all_algorithms()
                analyzer.generate_report()
            except Exception as e:
                print(f"⚠️  Analysis step failed: {e}")

            print("\nGenerating visualizations...")
            try:
                visualizer = ParkingVisualizer(str(results_path))
                visualizer.create_all_visualizations()

                # Save visualization artifacts
                import glob

                for viz_file in glob.glob("visualization_output/*.png"):
                    filename = viz_file.split("/")[-1]
                    save_artifact(
                        filename, source_path=viz_file, subdir="visualizations"
                    )
            except Exception as e:
                print(f"⚠️  Visualization step failed: {e}")

        elif args.mode == "analyze":
            # Run complexity analysis only
            print("Performing algorithmic complexity analysis...")
            # Create analysis directory in run folder
            analysis_dir = get_run_path() / "analysis"
            analysis_dir.mkdir(exist_ok=True)

            analyzer = ComplexityAnalyzer(output_dir=str(analysis_dir))
            analyzer.analyze_all_algorithms()
            analyzer.generate_report()

        elif args.mode == "visualize":
            # Create visualizations from existing results
            print("Creating visualizations...")
            visualizer = ParkingVisualizer(args.output)
            visualizer.create_all_visualizations()

            # Save visualization artifacts
            import glob

            for viz_file in glob.glob("visualization_output/*.png"):
                filename = viz_file.split("/")[-1]
                save_artifact(filename, source_path=viz_file, subdir="visualizations")

        # Complete the run tracking
        duration = time.time() - start_time
        complete_run("completed", duration)

        # Log completion and metrics
        main_logger.info("System execution completed")
        metrics.log_summary()

        print(f"\nCompleted at: {datetime.now()}")
        print(f"📁 Run artifacts saved to: output/runs/{run_id}")
        print(f"⏱️  Total duration: {duration:.1f} seconds")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        # Complete run with interrupted status
        try:
            duration = time.time() - start_time
            complete_run("interrupted", duration)
        except:
            pass
        main_logger.warning("System interrupted by user")
        print("\n⚠️  System interrupted by user")
        return 1
    except Exception as e:
        # Complete run with failed status
        try:
            duration = time.time() - start_time
            complete_run("failed", duration)
        except:
            pass
        main_logger.error(f"System execution failed: {e}")
        print(f"\n❌ System execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
