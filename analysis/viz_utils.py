"""
Base visualization utilities for parking optimization analysis and reporting.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

# Default dark theme settings matching PlantDoc style
DEFAULT_THEME = {
    "background_color": "#121212",
    "text_color": "#f5f5f5",
    "grid_color": "#404040",
    "main_color": "#34d399",
    "bar_colors": ["#a78bfa", "#22d3ee", "#34d399", "#f59e0b", "#ef4444"],
    "cmap": "YlOrRd",
}


def apply_dark_theme(theme: Optional[Dict] = None) -> None:
    """
    Apply dark theme to matplotlib plots.

    Args:
        theme: Theme settings dictionary (uses DEFAULT_THEME if None)
    """
    theme = theme or DEFAULT_THEME

    # Set the style
    plt.style.use("dark_background")

    # Configure plot settings
    plt.rcParams["figure.facecolor"] = theme["background_color"]
    plt.rcParams["axes.facecolor"] = theme["background_color"]
    plt.rcParams["text.color"] = theme["text_color"]
    plt.rcParams["axes.labelcolor"] = theme["text_color"]
    plt.rcParams["xtick.color"] = theme["text_color"]
    plt.rcParams["ytick.color"] = theme["text_color"]
    plt.rcParams["grid.color"] = theme["grid_color"]
    plt.rcParams["axes.edgecolor"] = theme["grid_color"]
    plt.rcParams["savefig.facecolor"] = theme["background_color"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", theme["bar_colors"])

    # Improve font rendering
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    # Better line and marker settings
    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2

    # Configure seaborn
    sns.set_palette(theme["bar_colors"])
    sns.set_style(
        "darkgrid",
        {
            "axes.facecolor": theme["background_color"],
            "grid.color": theme["grid_color"],
            "grid.alpha": 0.3,
        },
    )
    sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2.0})

    print(f"Applied dark theme with {theme['background_color']} background")


def ensure_plots_dir(output_dir: Path) -> Path:
    """Ensure plots directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_theme_colors() -> Dict[str, str]:
    """Get current theme colors for manual color assignment."""
    return DEFAULT_THEME


def create_custom_colormap(colors: Optional[list] = None):
    """Create a custom colormap from theme colors."""
    colors = DEFAULT_THEME["bar_colors"]

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("custom", colors)


def format_axis_labels(
    ax,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Apply consistent formatting to axis labels with theme colors."""
    theme = DEFAULT_THEME

    if title:
        ax.set_title(
            title, color=theme["text_color"], fontsize=16, fontweight="bold", pad=20
        )
    if xlabel:
        ax.set_xlabel(xlabel, color=theme["text_color"], fontsize=13)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme["text_color"], fontsize=13)

    # Style the spines
    for spine in ax.spines.values():
        spine.set_color(theme["grid_color"])
        spine.set_linewidth(1.2)

    # Style the ticks
    ax.tick_params(colors=theme["text_color"], labelsize=11)

    # Add subtle background for better readability
    ax.patch.set_facecolor(theme["background_color"])


def format_legend(ax, **kwargs):
    """Apply consistent legend formatting."""
    theme = DEFAULT_THEME
    legend = ax.legend(**kwargs)

    if legend:
        # Set legend background and border
        legend.get_frame().set_facecolor(theme["background_color"])
        legend.get_frame().set_edgecolor(theme["grid_color"])
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_linewidth(1.2)

        # Set text colors
        for text in legend.get_texts():
            text.set_color(theme["text_color"])
            text.set_fontsize(11)


def save_plot(fig, filepath: str, dpi: int = 300):
    """Save plot with consistent dark theme background."""
    theme = DEFAULT_THEME
    fig.patch.set_facecolor(theme["background_color"])

    plt.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=theme["background_color"],
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)
    print(f"Saved plot: {filepath}")
