"""
Run Management System
Creates organized folders for each simulation run with timestamp tracking
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RunMetadata:
    """Metadata for a simulation run"""

    run_id: str
    timestamp: str
    mode: str
    parameters: Dict
    duration_seconds: float = 0.0
    status: str = "running"  # running, completed, failed
    artifacts: List[str] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


class RunManager:
    """
    Manages simulation runs and organizes all artifacts
    """

    def __init__(self, base_output_dir: str = "output"):
        self.base_output_dir = Path(base_output_dir)
        self.runs_dir = self.base_output_dir / "runs"
        self.latest_link = self.base_output_dir / "latest"

        # Ensure directories exist
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.current_run: Optional[RunMetadata] = None
        self.current_run_dir: Optional[Path] = None

    def start_run(self, mode: str, parameters: Optional[Dict] = None) -> str:
        """
        Start a new simulation run

        Args:
            mode: Simulation mode (demo, simulate, analyze, etc.)
            parameters: Run parameters (drivers, duration, etc.)

        Returns:
            Run ID string
        """
        # Generate timestamp-based run ID
        timestamp = datetime.now()
        run_id = f"run_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"

        # Create run directory
        self.current_run_dir = self.runs_dir / run_id
        self.current_run_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.current_run_dir / "logs").mkdir(exist_ok=True)
        (self.current_run_dir / "visualizations").mkdir(exist_ok=True)
        (self.current_run_dir / "analysis").mkdir(exist_ok=True)

        # Initialize metadata
        self.current_run = RunMetadata(
            run_id=run_id,
            timestamp=timestamp.isoformat(),
            mode=mode,
            parameters=parameters or {},
            status="running",
        )

        # Save initial metadata
        self._save_metadata()

        logger.info(f"Started run {run_id} in mode '{mode}'")
        logger.info(f"Run directory: {self.current_run_dir}")

        return run_id

    def get_run_path(self, subdir: str = "") -> Path:
        """
        Get path for saving artifacts in current run

        Args:
            subdir: Subdirectory within run (logs, visualizations, analysis)

        Returns:
            Path object for saving files
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call start_run() first.")

        if subdir:
            path = self.current_run_dir / subdir
            path.mkdir(exist_ok=True)
            return path

        return self.current_run_dir

    def save_artifact(
        self,
        filename: str,
        content: Optional[str] = None,
        source_path: Optional[str] = None,
        subdir: str = "",
    ) -> Path:
        """
        Save an artifact to the current run directory

        Args:
            filename: Name of the file to save
            content: String content to write (if creating new file)
            source_path: Path to existing file to copy
            subdir: Subdirectory to save in

        Returns:
            Path where file was saved
        """
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        target_path = self.get_run_path(subdir) / filename

        if content is not None:
            # Save string content
            target_path.write_text(content)
        elif source_path:
            # Copy existing file
            source = Path(source_path)
            if source.exists():
                shutil.copy2(source, target_path)
            else:
                logger.warning(f"Source file not found: {source_path}")
                return target_path
        else:
            raise ValueError("Must provide either content or source_path")

        # Track artifact
        relative_path = str(target_path.relative_to(self.current_run_dir))
        if relative_path not in self.current_run.artifacts:
            self.current_run.artifacts.append(relative_path)
            self._save_metadata()

        logger.debug(f"Saved artifact: {relative_path}")
        return target_path

    def complete_run(self, status: str = "completed", duration_seconds: float = 0.0):
        """
        Mark current run as completed and update latest symlink

        Args:
            status: Final status (completed, failed)
            duration_seconds: Total run duration
        """
        if not self.current_run:
            logger.warning("No active run to complete")
            return

        self.current_run.status = status
        self.current_run.duration_seconds = duration_seconds
        self._save_metadata()

        # Update latest symlink
        if self.latest_link.is_symlink() or self.latest_link.exists():
            self.latest_link.unlink()

        # Create relative symlink
        relative_path = os.path.relpath(self.current_run_dir, self.base_output_dir)
        self.latest_link.symlink_to(relative_path)

        logger.info(f"Completed run {self.current_run.run_id} with status: {status}")
        logger.info(f"Artifacts saved to: {self.current_run_dir}")
        logger.info(f"Latest run link updated: {self.latest_link}")

        # Clear current run
        self.current_run = None
        self.current_run_dir = None

    def _save_metadata(self):
        """Save run metadata to JSON file"""
        if self.current_run and self.current_run_dir:
            metadata_path = self.current_run_dir / "metadata.json"
            metadata_path.write_text(json.dumps(asdict(self.current_run), indent=2))

    def list_runs(self) -> List[Dict]:
        """
        List all previous runs with their metadata

        Returns:
            List of run metadata dictionaries
        """
        runs = []

        for run_dir in sorted(self.runs_dir.iterdir()):
            if run_dir.is_dir():
                metadata_file = run_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text())
                        runs.append(metadata)
                    except Exception as e:
                        logger.warning(
                            f"Could not read metadata for {run_dir.name}: {e}"
                        )

        return runs

    def get_run_summary(self) -> str:
        """
        Get a summary of all runs

        Returns:
            Formatted string summary
        """
        runs = self.list_runs()

        if not runs:
            return "No runs found."

        summary = f"\n{'=' * 60}\n"
        summary += f"SIMULATION RUNS SUMMARY ({len(runs)} total)\n"
        summary += f"{'=' * 60}\n\n"

        for run in runs[-10:]:  # Show last 10 runs
            timestamp = datetime.fromisoformat(run["timestamp"])
            duration = run.get("duration_seconds", 0)
            status = run.get("status", "unknown")

            status_icon = {"completed": "✅", "failed": "❌", "running": "🔄"}.get(
                status, "❓"
            )

            summary += f"{status_icon} {run['run_id']}\n"
            summary += f"   📅 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"   🎯 Mode: {run['mode']}\n"
            summary += f"   ⏱️  Duration: {duration:.1f}s\n"
            summary += f"   📁 Artifacts: {len(run.get('artifacts', []))}\n"

            if run.get("parameters"):
                params = run["parameters"]
                if "drivers" in params:
                    summary += f"   🚗 Drivers: {params['drivers']}\n"
                if "duration" in params:
                    summary += f"   🕐 Sim Duration: {params['duration']}h\n"

            summary += "\n"

        # Show latest run path
        if self.latest_link.exists():
            summary += f"📍 Latest run: {self.latest_link.resolve()}\n"

        return summary

    def cleanup_old_runs(self, keep_count: int = 10):
        """
        Clean up old runs, keeping only the most recent ones

        Args:
            keep_count: Number of recent runs to keep
        """
        runs = self.list_runs()

        if len(runs) <= keep_count:
            logger.info(f"Only {len(runs)} runs found, no cleanup needed")
            return

        # Sort by timestamp and get old runs
        runs_sorted = sorted(runs, key=lambda x: x["timestamp"])
        old_runs = runs_sorted[:-keep_count]

        removed_count = 0
        for run in old_runs:
            run_dir = self.runs_dir / run["run_id"]
            if run_dir.exists():
                shutil.rmtree(run_dir)
                removed_count += 1
                logger.info(f"Removed old run: {run['run_id']}")

        logger.info(
            f"Cleaned up {removed_count} old runs, kept {keep_count} recent runs"
        )


# Global run manager instance
_run_manager: Optional[RunManager] = None


def get_run_manager() -> RunManager:
    """Get global run manager instance"""
    global _run_manager
    if _run_manager is None:
        _run_manager = RunManager()
    return _run_manager


def start_run(mode: str, parameters: Optional[Dict] = None) -> str:
    """Start a new simulation run"""
    return get_run_manager().start_run(mode, parameters)


def get_run_path(subdir: str = "") -> Path:
    """Get path for saving artifacts in current run"""
    return get_run_manager().get_run_path(subdir)


def save_artifact(
    filename: str,
    content: Optional[str] = None,
    source_path: Optional[str] = None,
    subdir: str = "",
) -> Path:
    """Save an artifact to the current run directory"""
    return get_run_manager().save_artifact(filename, content, source_path, subdir)


def complete_run(status: str = "completed", duration_seconds: float = 0.0):
    """Complete the current run"""
    get_run_manager().complete_run(status, duration_seconds)
