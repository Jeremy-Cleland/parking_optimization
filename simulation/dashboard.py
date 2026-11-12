"""Local operator UI; experiments run in the existing Python environment."""

import json
import mimetypes
import shutil
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import mkdtemp
from urllib.parse import unquote, urlparse

from simulation.experiments import ExperimentConfig, run_experiment, write_report


def config_from_request(payload: dict) -> ExperimentConfig:
    """Validate the bounded local UI workload independently of HTML controls."""
    allowed = {
        "seeds",
        "n_zones",
        "n_drivers",
        "duration_hours",
        "time_step_minutes",
        "scenario_event",
        "demand_multiplier",
        "min_price",
        "max_price",
        "target_occupancy",
        "max_price_change",
        "start_time",
        "warmup_hours",
        "array_backend",
    }
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unknown scenario setting")
    values = dict(payload)
    seeds = values.get("seeds", [1, 2, 3, 4, 5])
    if not isinstance(seeds, list) or not 2 <= len(seeds) <= 20:
        raise ValueError("Select 2-20 distinct seeds")
    values["seeds"] = tuple(seeds)
    for field in ("n_zones", "n_drivers"):
        if field in values and (
            not isinstance(values[field], int) or isinstance(values[field], bool)
        ):
            raise ValueError(f"{field} must be an integer")
    if (
        not 1 <= values.get("n_zones", 8) <= 100
        or not 0 <= values.get("n_drivers", 100) <= 2000
    ):
        raise ValueError("Use 1-100 zones and 0-2000 drivers")
    if (
        not 0 < values.get("duration_hours", 2) <= 12
        or not 1 <= values.get("time_step_minutes", 5) <= 60
    ):
        raise ValueError("Use a duration up to 12 hours and a step of 1-60 minutes")
    if not 0 < values.get("demand_multiplier", 1) <= 3:
        raise ValueError("Demand multiplier must be in (0, 3]")
    # Bound the product of settings, not only each slider separately.
    work = (
        len(seeds)
        * 4
        * values.get("duration_hours", 2)
        * 60
        / values.get("time_step_minutes", 5)
        * max(1, values.get("n_drivers", 100))
        * values.get("demand_multiplier", 1)
        * values.get("n_zones", 8)
    )
    if work > 20_000_000:
        raise ValueError(
            "Scenario is too large for the local screen; reduce drivers, zones, seeds or duration, or use compare mode in the CLI"
        )
    if values.get("scenario_event", "normal") not in {"normal", "event", "steady"}:
        raise ValueError("Unknown event scenario")
    if "start_time" in values:
        values["start_time"] = datetime.fromisoformat(values["start_time"])
    return ExperimentConfig(**values)


def make_handler(output_dir: Path) -> type[BaseHTTPRequestHandler]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server: HTTPServer
        latest_directory = root / "saved-comparison"

        def reply(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self.reply(200, template, "text/html; charset=utf-8")
                return
            if path == "/latest":
                artifact = type(self).latest_directory / "experiment.json"
                if artifact.is_file():
                    self.reply(
                        200,
                        json.dumps(
                            {
                                "report": json.loads(artifact.read_text()),
                                "artifact_base": f"/artifacts/{artifact.parent.name}",
                            }
                        ).encode(),
                        "application/json",
                    )
                else:
                    self.reply(200, b"null", "application/json")
                return
            if path.startswith("/artifacts/"):
                target = (root / path.removeprefix("/artifacts/")).resolve()
                if (
                    target.is_relative_to(root)
                    and target.is_file()
                    and target.suffix in {".json", ".csv", ".md", ".log", ".png"}
                ):
                    self.reply(
                        200,
                        target.read_bytes(),
                        mimetypes.guess_type(target.name)[0]
                        or "text/plain; charset=utf-8",
                    )
                    return
            self.reply(404, b"Artifact not found", "text/plain")

        def do_POST(self) -> None:
            allowed_origins = {
                f"http://{host}:{self.server.server_port}"
                for host in ("127.0.0.1", "localhost")
            }
            if (
                self.path != "/compare"
                or self.headers.get("Origin") not in allowed_origins
                or self.headers.get("Content-Type") != "application/json"
            ):
                self.reply(
                    403,
                    json.dumps(
                        {"error": "Use the local scenario screen to run a comparison"}
                    ).encode(),
                    "application/json",
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10000:
                    raise ValueError("Invalid request size")
                config = config_from_request(json.loads(self.rfile.read(length)))
                directory = Path(mkdtemp(prefix="experiment-", dir=root))
                report = run_experiment(config, directory)
                type(self).latest_directory = directory
                response = {
                    "report": report,
                    "artifact_base": f"/artifacts/{directory.name}",
                }
                self.reply(
                    200,
                    json.dumps(response, allow_nan=False).encode(),
                    "application/json",
                )
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(
                    400, json.dumps({"error": str(exc)}).encode(), "application/json"
                )
            except Exception as exc:
                self.reply(
                    500,
                    json.dumps(
                        {
                            "error": f"Experiment failed ({type(exc).__name__}); see the local terminal and saved run logs"
                        }
                    ).encode(),
                    "application/json",
                )

    return Handler


def restore_latest_comparison(output_dir: Path) -> bool:
    """Archive the last comparison so it can be viewed without rerunning it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = output_dir / "saved-comparison"
    if saved.exists():
        write_report(json.loads((saved / "experiment.json").read_text()), saved)
        return True
    runs_dir = output_dir.parent.parent
    candidates = sorted(
        [
            *runs_dir.glob("run_*/comparison/experiment.json"),
            *runs_dir.glob("run_*/experiments/experiment-*/experiment.json"),
        ],
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for candidate in candidates:
        try:
            report = json.loads(candidate.read_text())
            if report.get("scope") != "simulation_only":
                continue
        except (OSError, ValueError, KeyError):
            continue
        shutil.copytree(candidate.parent, saved)
        write_report(report, saved)
        return True
    return False


def serve_dashboard(output_dir: Path, port: int = 8765) -> None:
    # A newly opened dashboard shows the last completed comparison immediately.
    restore_latest_comparison(output_dir)
    with HTTPServer(("127.0.0.1", port), make_handler(output_dir)) as server:
        print(
            f"Local scenario dashboard: http://127.0.0.1:{server.server_port}",
            flush=True,
        )
        print(
            "Press Ctrl+C to stop. Experiments and reports remain on disk.", flush=True
        )
        server.serve_forever()
