"""Render policy comparisons directly from the report's recorded estimates."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

PANELS = (
    ("revenue", "Occupied-space revenue", "USD", 1),
    ("mean_occupancy", "Occupied capacity", "Percent of capacity", 100),
    (
        "success_rate",
        "Drivers who parked",
        "Percent of measurement-window arrivals",
        100,
    ),
    ("mean_stay_cost", "Quoted full-stay cost", "USD per successful driver", 1),
)


def comparison_figure(report: dict) -> Figure:
    """Show means and seed confidence intervals without inventing missing values."""
    background, text, muted = "#111113", "#f8fafc", "#cbd5e1"
    figure = Figure(figsize=(13, 8.5), facecolor=background, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).flat
    policies = report["config"]["policies"]
    seeds = len(report["config"]["seeds"])
    figure.suptitle(
        f"Simulated policy comparison | {seeds} scenario seeds | {report['config'].get('warmup_hours', 0):g}h warm-up excluded\n"
        "Dots: policy means. Lines: 95% confidence intervals across seeds.",
        color=text,
        fontsize=15,
        linespacing=1.6,
    )
    for axis, (metric, title, unit, scale) in zip(axes, PANELS, strict=True):
        axis.set_facecolor(background)
        for index, policy in enumerate(policies):
            estimate = report["summary"][policy]["metrics"][metric]
            if estimate["mean"] is None:
                axis.text(
                    0.03,
                    index,
                    "No observations",
                    color=muted,
                    transform=axis.get_yaxis_transform(),
                    va="center",
                )
                continue
            mean = estimate["mean"] * scale
            interval = estimate["ci95"]
            error = (
                None
                if interval is None
                else [
                    [max(0, mean - interval[0] * scale)],
                    [max(0, interval[1] * scale - mean)],
                ]
            )
            color = "#34d399" if policy.startswith("dynamic") else "#60a5fa"
            axis.errorbar(
                mean,
                index,
                xerr=error,
                fmt="o",
                color=color,
                capsize=4,
                markersize=6,
                linewidth=1.7,
            )
        axis.set_yticks(range(len(policies)), [p.replace("-", " + ") for p in policies])
        axis.set_ylim(len(policies) - 0.4, -0.6)
        axis.set_title(title, loc="left", color=text, fontsize=13, pad=14)
        axis.set_xlabel(unit, color=muted, labelpad=9)
        axis.tick_params(colors=muted, labelsize=10)
        axis.grid(axis="x", color="#334155", alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#334155")
        axis.margins(x=0.15)
    figure.supxlabel(
        "Cost is conditional on successful parking. Intervals omit model uncertainty.",
        fontsize=10,
        color=muted,
    )
    return figure


def save_comparison_chart(report: dict, path: Path) -> None:
    figure = comparison_figure(report)
    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())
