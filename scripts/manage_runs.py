#!/usr/bin/env python3
"""
Run Management CLI
Manage simulation runs and artifacts
"""

import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.run_manager import get_run_manager


def list_runs():
    """List all simulation runs"""
    manager = get_run_manager()
    summary = manager.get_run_summary()
    print(summary)


def show_run(run_id: str):
    """Show details of a specific run"""
    manager = get_run_manager()
    runs = manager.list_runs()

    target_run = None
    for run in runs:
        if run["run_id"] == run_id:
            target_run = run
            break

    if not target_run:
        print(f"❌ Run '{run_id}' not found")
        return

    print(f"\n{'=' * 60}")
    print(f"RUN DETAILS: {run_id}")
    print(f"{'=' * 60}")

    print(f"📅 Timestamp: {target_run['timestamp']}")
    print(f"🎯 Mode: {target_run['mode']}")
    print(f"⏱️  Duration: {target_run.get('duration_seconds', 0):.1f}s")
    print(f"📊 Status: {target_run.get('status', 'unknown')}")

    if target_run.get("parameters"):
        print("\n📋 Parameters:")
        for key, value in target_run["parameters"].items():
            print(f"   {key}: {value}")

    if target_run.get("artifacts"):
        print(f"\n📁 Artifacts ({len(target_run['artifacts'])}):")
        for artifact in target_run["artifacts"]:
            print(f"   📄 {artifact}")

    # Show actual directory contents
    run_dir = manager.runs_dir / run_id
    if run_dir.exists():
        print("\n📂 Directory Contents:")
        for item in sorted(run_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(run_dir)
                size_kb = item.stat().st_size / 1024
                print(f"   📄 {rel_path} ({size_kb:.1f}KB)")


def cleanup_runs(keep_count: int = 10):
    """Clean up old runs"""
    manager = get_run_manager()
    manager.cleanup_old_runs(keep_count)
    print(f"✅ Cleanup completed, kept {keep_count} most recent runs")


def open_run(run_id: str):
    """Open run directory in file browser"""
    manager = get_run_manager()
    run_dir = manager.runs_dir / run_id

    if not run_dir.exists():
        print(f"❌ Run '{run_id}' not found")
        return

    import platform
    import subprocess

    system = platform.system()
    if system == "Darwin":  # macOS
        subprocess.run(["open", str(run_dir)])
    elif system == "Windows":
        subprocess.run(["explorer", str(run_dir)])
    elif system == "Linux":
        subprocess.run(["xdg-open", str(run_dir)])
    else:
        print(f"📁 Run directory: {run_dir}")


def compare_runs(run1_id: str, run2_id: str):
    """Compare two simulation runs"""
    manager = get_run_manager()
    runs = {run["run_id"]: run for run in manager.list_runs()}

    if run1_id not in runs:
        print(f"❌ Run '{run1_id}' not found")
        return
    if run2_id not in runs:
        print(f"❌ Run '{run2_id}' not found")
        return

    run1 = runs[run1_id]
    run2 = runs[run2_id]

    print(f"\n{'=' * 60}")
    print(f"RUN COMPARISON: {run1_id} vs {run2_id}")
    print(f"{'=' * 60}")

    print(f"{'Metric':<20} {'Run 1':<20} {'Run 2':<20} {'Difference'}")
    print(f"{'-' * 20} {'-' * 20} {'-' * 20} {'-' * 10}")

    # Compare basic metrics
    metrics = [
        ("Mode", "mode", str),
        ("Duration (s)", "duration_seconds", lambda x: f"{x:.1f}"),
        ("Status", "status", str),
        ("Artifacts", lambda r: len(r.get("artifacts", [])), str),
    ]

    for name, key, formatter in metrics:
        if callable(key):
            val1 = key(run1)
            val2 = key(run2)
        else:
            val1 = run1.get(key, "N/A")
            val2 = run2.get(key, "N/A")

        val1_str = formatter(val1) if val1 != "N/A" else "N/A"
        val2_str = formatter(val2) if val2 != "N/A" else "N/A"

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            diff_str = f"{diff:+.1f}" if abs(diff) > 0.1 else "≈0"
        else:
            diff_str = "✓" if val1 == val2 else "≠"

        print(f"{name:<20} {val1_str:<20} {val2_str:<20} {diff_str}")

    # Compare parameters
    params1 = run1.get("parameters", {})
    params2 = run2.get("parameters", {})

    if params1 or params2:
        print("\n📋 Parameter Comparison:")
        all_params = set(params1.keys()) | set(params2.keys())
        for param in sorted(all_params):
            p1 = params1.get(param, "N/A")
            p2 = params2.get(param, "N/A")
            diff_str = "✓" if p1 == p2 else "≠"
            print(f"   {param:<15} {p1:<15} {p2:<15} {diff_str}")


def main():
    parser = argparse.ArgumentParser(description="Manage simulation runs")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    subparsers.add_parser("list", help="List all runs")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("run_id", help="Run ID to show")

    # Open command
    open_parser = subparsers.add_parser("open", help="Open run directory")
    open_parser.add_argument("run_id", help="Run ID to open")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old runs")
    cleanup_parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="Number of recent runs to keep (default: 10)",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two runs")
    compare_parser.add_argument("run1", help="First run ID")
    compare_parser.add_argument("run2", help="Second run ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "list":
        list_runs()
    elif args.command == "show":
        show_run(args.run_id)
    elif args.command == "open":
        open_run(args.run_id)
    elif args.command == "cleanup":
        cleanup_runs(args.keep)
    elif args.command == "compare":
        compare_runs(args.run1, args.run2)


if __name__ == "__main__":
    main()
