"""Export and verify the compact public evidence, without publishing raw map data.

Run from the repository root: python -m scripts.export_evaluation STUDY VALIDATION.
Use --verify to check the published seed outcomes and aggregates without raw runs.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from simulation.experiments import aggregate

DESTINATION = Path("docs/evaluation")
LEVELS = ("low", "medium", "high")


def consistent(actual, expected) -> bool:
    """Allow rounding differences in SciPy/NumPy across supported platforms."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            consistent(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            consistent(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9)
    return actual == expected


def verify(destination: Path = DESTINATION) -> dict:
    evidence = json.loads((destination / "evidence.json").read_text())
    for studies in evidence["studies"].values():
        for report in studies.values():
            calculated = aggregate(report["runs"], report["config"]["policies"])
            if not consistent(calculated, report["summary"]):
                raise ValueError("Published aggregates do not match seed outcomes")
            for seed in report["config"]["seeds"]:
                runs = [r for r in report["runs"] if r["seed"] == seed]
                if len({r["scenario_fingerprint"] for r in runs}) != 1:
                    raise ValueError("Unmatched policy inputs")
    for name, digest in evidence["artifact_sha256"].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Public artifact hash mismatch: {name}")
    return evidence


def export(discovery: Path, validation: Path) -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    evidence = {"scope": "simulation_only", "date": "2026-11-12", "studies": {}}
    fieldnames = None
    csv_rows = []
    source_hashes = None
    for name, folder in (("original", discovery), ("validation", validation)):
        reports = {}
        for level in LEVELS:
            source = folder / ("comparison" if level == "high" else level)
            report = json.loads((source / "experiment.json").read_text())
            for run in report["runs"]:
                raw = source / run["artifact"]
                if hashlib.sha256(raw.read_bytes()).hexdigest() != run["sha256"]:
                    raise ValueError("Raw result hash mismatch")
                result = json.loads(raw.read_text())
                if result["simulation_config"]["billing_policy"] != "entry_rate":
                    raise ValueError("Evaluation requires entry-rate billing")
                if source_hashes is None:
                    source_hashes = report["source_sha256"]
                    evidence["data_validation"] = result["data_validation"]
                if source_hashes != report["source_sha256"]:
                    raise ValueError("Algorithm changed between evaluation sets")
                row = {
                    "study": name,
                    "demand": level,
                    "seed": run["seed"],
                    "policy": run["policy"],
                    **run["outcomes"],
                }
                fieldnames = list(row)
                csv_rows.append(row)
            reports[level] = report
        evidence["studies"][name] = reports
        shutil.copyfile(
            folder / "demand_study.png", DESTINATION / f"{name}_demand_study.png"
        )
    shutil.copyfile(validation / "demand_study.png", DESTINATION / "demand_study.png")
    shutil.copyfile(
        validation / "comparison/policy_comparison.png", DESTINATION / "high_demand.png"
    )
    with (DESTINATION / "seed_outcomes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    evidence["artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(DESTINATION.iterdir())
        if p.suffix in {".csv", ".png"}
    }
    (DESTINATION / "evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    verify()
    for name, reports in evidence["studies"].items():
        print(name)
        for level, report in reports.items():
            base = report["summary"]["static-distance"]["metrics"]
            dyn = report["summary"]["dynamic-distance"]
            diff = dyn["paired_difference_from_baseline"]
            m = dyn["metrics"]
            print(
                level,
                "revenue",
                base["revenue"]["mean"],
                m["revenue"]["mean"],
                "relative",
                (m["revenue"]["mean"] / base["revenue"]["mean"] - 1) * 100,
                "deltaCI",
                diff["revenue"]["ci95"],
                "occupancy",
                base["mean_occupancy"]["mean"],
                m["mean_occupancy"]["mean"],
                "success",
                base["success_rate"]["mean"],
                m["success_rate"]["mean"],
                "successCI",
                [v * 100 for v in diff["success_rate"]["ci95"]],
                "cost",
                base["mean_stay_cost"]["mean"],
                m["mean_stay_cost"]["mean"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("discovery", nargs="?", type=Path)
    parser.add_argument("validation", nargs="?", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        print(
            "Published seed outcomes, paired aggregates and artifact hashes verified."
        )
    elif args.discovery and args.validation:
        export(args.discovery, args.validation)
    else:
        parser.error("Supply both study folders, or --verify")
