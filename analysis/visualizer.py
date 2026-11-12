"""
Visualization Module for Parking Optimization Results
Creates charts and maps to visualize simulation outcomes
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .viz_utils import (
    apply_dark_theme,
    format_axis_labels,
    format_legend,
    get_theme_colors,
    save_plot,
)


class ParkingVisualizer:
    """
    Creates visualizations for parking optimization results
    """

    def __init__(
        self,
        results_file: str = "output/results.json",
        output_dir: str = "visualization_output",
    ):
        """
        Initialize visualizer with results file

        Args:
            results_file: Path to simulation results JSON
        """
        self.results_file = results_file
        self.results = None
        self.output_dir = str(output_dir)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Apply dark theme
        apply_dark_theme()
        self.theme_colors = get_theme_colors()

        # Load results
        self._load_results()

    def _load_results(self):
        """Load simulation results from JSON file"""
        try:
            with open(self.results_file) as f:
                self.results = json.load(f)
            print(f"Loaded results from {self.results_file}")
        except FileNotFoundError:
            print(f"Results file {self.results_file} not found. Run simulation first.")
            self.results = None

    def _context_title(self, title):
        config = self.results.get("simulation_config", {})
        policy = f"{config.get('pricing_policy', 'legacy')}/{config.get('routing_policy', 'legacy')}"
        return f"{title}\nSimulated | {policy} | seed {config.get('seed', 'unknown')}"

    def _outcome_counts(self):
        metrics = self.results["metrics"]
        parked, rejected = metrics["successful_parks"], metrics["rejected_drivers"]
        unresolved = self.results.get("driver_outcomes", {}).get("unresolved")
        if unresolved is None:
            total = self.results.get("simulation_config", {}).get(
                "n_drivers", parked + rejected
            )
            unresolved = max(0, total - parked - rejected)
        return [parked, rejected, unresolved]

    def _elapsed_minutes(self):
        """Use recorded interval ends, retaining support for older five-minute runs."""
        metrics = self.results["metrics"]
        step = self.results.get("simulation_config", {}).get("time_step_minutes", 5)
        return metrics.get(
            "elapsed_minutes",
            [(i + 1) * step for i in range(len(metrics["avg_occupancy"]))],
        )

    def _hourly_revenue(self):
        """Allocate each recorded interval to elapsed-hour bins, including partial hours."""
        totals = {}
        start = 0
        for end, revenue in zip(
            self._elapsed_minutes(),
            self.results["metrics"]["total_revenue"],
            strict=True,
        ):
            cursor = start
            while cursor < end:
                hour = int(cursor // 60)
                boundary = min(end, (hour + 1) * 60)
                totals[hour] = totals.get(hour, 0) + revenue * (boundary - cursor) / (
                    end - start
                )
                cursor = boundary
            start = end
        return list(totals.values())

    def create_all_visualizations(self):
        """Create all visualization types"""
        if not self.results:
            print("No results to visualize.")
            return

        print("\nCreating visualizations...")

        # 1. Performance metrics over time
        self.plot_performance_metrics()

        # 2. Revenue analysis
        self.plot_revenue_analysis()

        # 3. Search time distribution
        self.plot_search_time_distribution()

        # 4. Occupancy heatmap
        self.plot_occupancy_heatmap()

        # 5. Algorithm performance comparison
        self.plot_algorithm_comparison()

        # 6. Create summary dashboard
        self.create_dashboard()

        print(f"\nVisualizations saved to '{self.output_dir}/' directory")

    def plot_performance_metrics(self):
        """Plot key performance metrics over time"""
        metrics = self.results["metrics"]

        # Create time axis (5-minute intervals)
        time_steps = self._elapsed_minutes()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            self._context_title("Parking Policy Observations"),
            fontsize=20,
            color=self.theme_colors["text_color"],
            y=0.95,
        )

        # 1. Occupancy over time
        ax = axes[0, 0]
        ax.plot(
            time_steps,
            np.array(metrics["avg_occupancy"]) * 100,
            color=self.theme_colors["main_color"],
            linewidth=3,
            alpha=0.9,
        )
        ax.axhline(
            y=85,
            color=self.theme_colors["bar_colors"][3],
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Target (85%)",
        )
        format_axis_labels(
            ax, "Average Occupancy Rate", "Time (minutes)", "Occupancy (%)"
        )
        format_legend(ax, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        # 2. Revenue over time
        ax = axes[0, 1]
        cumulative_revenue = np.cumsum(metrics["total_revenue"])
        ax.plot(
            time_steps,
            cumulative_revenue,
            color=self.theme_colors["bar_colors"][2],
            linewidth=3,
            alpha=0.9,
        )
        ax.fill_between(
            time_steps,
            0,
            cumulative_revenue,
            color=self.theme_colors["bar_colors"][2],
            alpha=0.2,
        )
        format_axis_labels(
            ax, "Revenue Generation", "Time (minutes)", "Cumulative Revenue ($)"
        )
        ax.grid(True, alpha=0.3)

        # 3. Search time over time
        ax = axes[1, 0]
        if metrics["avg_search_time"]:
            # Create time steps for search times (they occur when drivers arrive)
            search_time_steps = list(range(len(metrics["avg_search_time"])))
            ax.plot(
                search_time_steps,
                metrics["avg_search_time"],
                color=self.theme_colors["bar_colors"][0],
                linewidth=2,
                alpha=0.8,
                marker="o",
                markersize=3,
                markevery=10,
            )
            format_axis_labels(
                ax,
                "Successful-Driver Search Delay",
                "Successful Parking Number",
                "Search Time (minutes)",
            )
            ax.grid(True, alpha=0.3)

        # 4. Success rate pie chart
        ax = axes[1, 1]
        total_drivers = metrics["successful_parks"] + metrics["rejected_drivers"]
        if total_drivers > 0:
            labels = ["Parked", "Rejected", "Unresolved"]
            sizes = self._outcome_counts()
            colors = [
                self.theme_colors["main_color"],
                self.theme_colors["bar_colors"][3],
                self.theme_colors["bar_colors"][1],
            ]

            _wedges, _texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": self.theme_colors["text_color"], "fontsize": 12},
                explode=tuple(
                    [0.05] + [0] * (len(sizes) - 1)
                ),  # slightly separate the slices
            )

            # Improve pie chart text styling
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(11)

            format_axis_labels(ax, f"Resolved Parking Outcomes (n={total_drivers})")

        # Remove top and right spines for cleaner look
        for ax_row in axes:
            for ax in ax_row:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        plt.tight_layout()
        save_plot(fig, f"{self.output_dir}/performance_metrics.png")

    def plot_revenue_analysis(self):
        """Analyze revenue generation patterns"""
        metrics = self.results["metrics"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(
            self._context_title("Occupied-Space Revenue"),
            fontsize=20,
            color=self.theme_colors["text_color"],
            y=0.95,
        )

        # 1. Revenue rate over time
        ax = axes[0]
        revenue_per_step = metrics["total_revenue"]
        time_steps = self._elapsed_minutes()

        ax.plot(
            time_steps,
            revenue_per_step,
            color=self.theme_colors["bar_colors"][1],
            alpha=0.6,
            linewidth=2,
            label="Raw",
        )

        # Moving average
        step = self.results.get("simulation_config", {}).get("time_step_minutes", 5)
        window = max(1, int(60 / step))
        if len(revenue_per_step) >= window:
            ma = pd.Series(revenue_per_step).rolling(window=window).mean()
            ax.plot(
                time_steps,
                ma,
                color=self.theme_colors["main_color"],
                linewidth=3,
                alpha=0.9,
                label=f"{window}-interval moving average",
            )

        format_axis_labels(
            ax,
            "Revenue by Simulation Interval",
            "Time (minutes)",
            "Revenue per interval ($)",
        )
        format_legend(ax, loc="best")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 2. Revenue by hour
        ax = axes[1]
        hourly_revenue = self._hourly_revenue()

        hours = list(range(len(hourly_revenue)))
        bars = ax.bar(
            hours,
            hourly_revenue,
            color=self.theme_colors["bar_colors"][2],
            alpha=0.8,
            edgecolor=self.theme_colors["text_color"],
            linewidth=0.5,
        )

        # Add value labels on bars
        for _i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(hourly_revenue) * 0.01,
                    f"${height:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=self.theme_colors["text_color"],
                    fontweight="bold",
                )

        format_axis_labels(ax, "Hourly Revenue Breakdown", "Hour", "Revenue ($)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add total revenue annotation with improved styling
        total_revenue = sum(metrics["total_revenue"])
        ax.text(
            0.02,
            0.98,
            f"Total: ${total_revenue:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            color=self.theme_colors["text_color"],
            fontsize=12,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": self.theme_colors["grid_color"],
                "alpha": 0.8,
                "edgecolor": self.theme_colors["main_color"],
                "linewidth": 1,
            },
        )

        plt.tight_layout()
        save_plot(fig, f"{self.output_dir}/revenue_analysis.png")

    def plot_search_time_distribution(self):
        """Plot distribution of driver search times"""
        metrics = self.results["metrics"]

        if not metrics["avg_search_time"]:
            print("No search time data available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create histogram of search times
        search_times = metrics["avg_search_time"]

        # Calculate optimal number of bins
        n_bins = min(30, max(10, len(search_times) // 10))

        _n, _bins, patches = ax.hist(
            search_times,
            bins=n_bins,
            color=self.theme_colors["main_color"],
            edgecolor=self.theme_colors["text_color"],
            alpha=0.8,
            linewidth=1.2,
            density=False,
        )

        # Color bars with gradient
        for i, patch in enumerate(patches):
            alpha = 0.6 + 0.4 * (i / len(patches))
            patch.set_alpha(alpha)

        # Add statistical lines
        mean_time = np.mean(search_times)
        median_time = np.median(search_times)

        ax.axvline(
            mean_time,
            color=self.theme_colors["bar_colors"][3],
            linestyle="--",
            linewidth=3,
            alpha=0.9,
            label=f"Mean: {mean_time:.1f} min",
        )
        ax.axvline(
            median_time,
            color=self.theme_colors["bar_colors"][1],
            linestyle="--",
            linewidth=3,
            alpha=0.9,
            label=f"Median: {median_time:.1f} min",
        )

        format_axis_labels(
            ax,
            "Distribution of Driver Search Times",
            "Search Time (minutes)",
            "Frequency",
        )
        format_legend(ax, loc="best")
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add comprehensive statistics box
        q25, q75 = np.percentile(search_times, [25, 75])
        stats_text = "Statistics\n"
        stats_text += f"{'─' * 15}\n"
        stats_text += f"Min: {np.min(search_times):.1f} min\n"
        stats_text += f"Q1: {q25:.1f} min\n"
        stats_text += f"Median: {median_time:.1f} min\n"
        stats_text += f"Q3: {q75:.1f} min\n"
        stats_text += f"Max: {np.max(search_times):.1f} min\n"
        stats_text += f"Mean: {mean_time:.1f} min\n"
        stats_text += f"Std Dev: {np.std(search_times):.1f} min\n"
        stats_text += f"IQR: {q75 - q25:.1f} min"

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            color=self.theme_colors["text_color"],
            fontsize=11,
            fontfamily="monospace",
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": self.theme_colors["grid_color"],
                "alpha": 0.9,
                "edgecolor": self.theme_colors["main_color"],
                "linewidth": 1.5,
            },
        )

        plt.tight_layout()
        save_plot(fig, f"{self.output_dir}/search_time_distribution.png")

    def plot_occupancy_heatmap(self):
        """Create occupancy heatmap over time"""
        metrics = self.results["metrics"]

        history = metrics.get("zone_occupancy")
        if not history:
            print("No recorded per-zone occupancy history; skipping heatmap.")
            return
        zone_ids = list(history[0])
        zone_occupancy = np.array(
            [[record[zone_id] for record in history] for zone_id in zone_ids]
        )
        n_zones, n_steps = zone_occupancy.shape

        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 10))

        # Use improved colormap with better dark theme compatibility
        from matplotlib.colors import LinearSegmentedColormap

        custom_colors = [
            "#0f172a",
            "#1e40af",
            "#22d3ee",
            "#34d399",
            "#fbbf24",
            "#f87171",
        ]
        custom_cmap = LinearSegmentedColormap.from_list(
            "occupancy", custom_colors, N=256
        )

        im = ax.imshow(
            zone_occupancy,
            cmap=custom_cmap,
            aspect="auto",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            origin="lower",
        )

        # Improve axis formatting
        # Set x-axis ticks every hour
        hour_indices = list(range(0, n_steps, max(1, n_steps // 8)))
        ax.set_xticks(hour_indices)
        ax.set_xticklabels(
            [f"{self._elapsed_minutes()[i] / 60:.2f}h" for i in hour_indices]
        )

        # Set y-axis with better zone labeling
        zone_step = max(1, n_zones // 20)  # Show at most 20 zone labels
        zone_indices = list(range(0, n_zones, zone_step))
        ax.set_yticks(zone_indices)
        ax.set_yticklabels([zone_ids[i] for i in zone_indices])

        format_axis_labels(ax, "Parking Zone Occupancy Heatmap", "Time", "Parking Zone")

        # Add improved colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label(
            "Occupancy Rate",
            rotation=270,
            labelpad=20,
            color=self.theme_colors["text_color"],
            fontsize=13,
            fontweight="bold",
        )
        cbar.ax.tick_params(colors=self.theme_colors["text_color"], labelsize=11)

        # Style the colorbar
        cbar.outline.set_edgecolor(self.theme_colors["grid_color"])
        cbar.outline.set_linewidth(1.2)

        # Add subtle grid for better readability
        ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_zones) - 0.5, minor=True)
        ax.grid(
            which="minor",
            color=self.theme_colors["grid_color"],
            linestyle="-",
            linewidth=0.3,
            alpha=0.4,
        )

        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        save_plot(fig, f"{self.output_dir}/occupancy_heatmap.png")

    def plot_algorithm_comparison(self):
        """Compare different algorithmic approaches"""
        complexity = self.results.get("algorithm_complexity", {})

        if not complexity:
            print("No algorithm complexity data available")
            # Create a static complexity comparison instead
            self._create_static_complexity_chart()
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract complexity classes
        algorithms = []
        complexities = []

        for module, data in complexity.items():
            for algo, comp in data.items():
                if "O(" in comp:
                    algorithms.append(f"{module}\n{algo}")
                    complexities.append(comp)

        # Create bar positions
        y_pos = np.arange(len(algorithms))

        # Create horizontal bar chart
        bars = ax.barh(y_pos, np.arange(len(algorithms)), alpha=0.8)

        # Color code by algorithm type using theme colors
        for i, bar in enumerate(bars):
            bar.set_color(
                self.theme_colors["bar_colors"][
                    i % len(self.theme_colors["bar_colors"])
                ]
            )

        # Add complexity annotations
        for i, (_algo, comp) in enumerate(zip(algorithms, complexities, strict=True)):
            ax.text(
                0.1,
                i,
                comp,
                va="center",
                fontweight="bold",
                color=self.theme_colors["text_color"],
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(algorithms)
        format_axis_labels(
            ax, "Algorithm Complexity Comparison", "Algorithmic Complexity"
        )
        ax.set_xlim(0, len(algorithms))

        # Remove x-axis as it's not meaningful
        ax.set_xticks([])

        # Add legend using theme colors
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][0],
                label="Routing (Shortest Paths)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][1],
                label="Pricing (Game Theory)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][2],
                label="Prediction (DP)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][3],
                label="Coordination (D&C)",
            ),
        ]
        ax.legend(handles=legend_elements, loc="right")

        plt.tight_layout()
        save_plot(fig, f"{self.output_dir}/algorithm_comparison.png")

    def create_dashboard(self):
        """Create a summary dashboard"""
        metrics = self.results["metrics"]
        config = self.results.get("simulation_config", {})

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            self._context_title("Parking Scenario Summary"),
            fontsize=24,
            color=self.theme_colors["text_color"],
            fontweight="bold",
            y=0.995,
        )

        # Create improved grid layout
        gs = fig.add_gridspec(
            3, 3, hspace=0.35, wspace=0.25, left=0.05, right=0.95, top=0.88, bottom=0.08
        )

        # 1. Key metrics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis("off")

        total_revenue = sum(metrics["total_revenue"])
        avg_occupancy = (
            float(
                np.average(
                    metrics["avg_occupancy"],
                    weights=np.diff([0, *self._elapsed_minutes()]),
                )
            )
            * 100
        )
        total_drivers = metrics["successful_parks"] + metrics["rejected_drivers"]
        success_rate = (
            (metrics["successful_parks"] / total_drivers * 100)
            if total_drivers > 0
            else 0
        )
        avg_search = (
            np.mean(metrics["avg_search_time"]) if metrics["avg_search_time"] else 0
        )

        capacity = sum(
            z.get("capacity", 0)
            for z in self.results.get("final_state", {}).get("zone_occupancy", [])
        )
        metrics_text = f"Scenario: {config.get('data_source', 'legacy')} | seed {config.get('seed', 'unknown')}\n"
        metrics_text += (
            f"Zones: {config.get('total_zones', 'N/A')} | spaces: {capacity:,}\n"
        )
        metrics_text += (
            f"Drivers: {config.get('n_drivers', 'N/A'):,}\n"
            if isinstance(config.get("n_drivers"), int)
            else "Drivers: unknown\n"
        )
        metrics_text += f"Duration: {config.get('duration_hours', '?')}h | warm-up: {config.get('warmup_hours', 0)}h\n\n"
        metrics_text += "Full run (includes warm-up)\n"
        metrics_text += f"Total Revenue: ${total_revenue:,.2f}\n"
        metrics_text += f"Avg Occupancy: {avg_occupancy:.1f}%\n"
        metrics_text += f"Resolved Success: {success_rate:.1f}%\n"
        metrics_text += f"Unresolved Drivers: {self._outcome_counts()[2]:,}\n"
        metrics_text += f"Successful Search: {avg_search:.1f} min\n"
        metrics_text += f"Search observations: {len(metrics['avg_search_time']):,}"

        ax1.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
            color=self.theme_colors["text_color"],
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": self.theme_colors["grid_color"],
                "alpha": 0.9,
                "edgecolor": self.theme_colors["main_color"],
                "linewidth": 2,
            },
        )

        # 2. Occupancy over time (top middle and right)
        ax2 = fig.add_subplot(gs[0, 1:])
        time_steps = [minute / 60 for minute in self._elapsed_minutes()]
        ax2.plot(
            time_steps,
            np.array(metrics["avg_occupancy"]) * 100,
            color=self.theme_colors["main_color"],
            linewidth=3,
            alpha=0.9,
        )
        ax2.axhline(
            y=85,
            color=self.theme_colors["bar_colors"][3],
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Target",
        )
        ax2.fill_between(
            time_steps,
            0,
            np.array(metrics["avg_occupancy"]) * 100,
            alpha=0.2,
            color=self.theme_colors["main_color"],
        )
        format_axis_labels(
            ax2, "Occupancy Rate Over Time", "Time (hours)", "Occupancy (%)"
        )
        format_legend(ax2, loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_ylim(0, 100)

        # 3. Revenue accumulation (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        cumulative_revenue = np.cumsum(metrics["total_revenue"])
        time_steps_min = self._elapsed_minutes()
        ax3.plot(
            time_steps_min,
            cumulative_revenue,
            color=self.theme_colors["bar_colors"][2],
            linewidth=3,
            alpha=0.9,
        )
        ax3.fill_between(
            time_steps_min,
            0,
            cumulative_revenue,
            alpha=0.3,
            color=self.theme_colors["bar_colors"][2],
        )
        format_axis_labels(ax3, "Cumulative Revenue", "Time (minutes)", "Revenue ($)")
        ax3.grid(True, alpha=0.3)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # 4. Search time trend (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if metrics["avg_search_time"]:
            search_time_ma = (
                pd.Series(metrics["avg_search_time"]).rolling(window=6).mean()
            )
            time_steps_search = list(range(1, len(metrics["avg_search_time"]) + 1))
            ax4.plot(
                time_steps_search,
                metrics["avg_search_time"],
                color=self.theme_colors["bar_colors"][0],
                alpha=0.5,
                linewidth=2,
                label="Raw",
            )
            ax4.plot(
                time_steps_search,
                search_time_ma,
                color=self.theme_colors["bar_colors"][0],
                linewidth=3,
                alpha=0.9,
                label="6-driver moving average",
            )
            format_axis_labels(
                ax4,
                "Driver Search Time",
                "Successful Parking Number",
                "Search Time (min)",
            )
            format_legend(ax4, loc="best")
            ax4.grid(True, alpha=0.3)
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)

        # 5. Success/Rejection pie (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        sizes = self._outcome_counts()
        if sum(sizes) > 0:
            labels = ["Parked", "Rejected", "Unresolved"]
            colors = [
                self.theme_colors["main_color"],
                self.theme_colors["bar_colors"][3],
                self.theme_colors["bar_colors"][1],
            ]
            _wedges, _texts, autotexts = ax5.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": self.theme_colors["text_color"], "fontsize": 11},
                explode=tuple([0.05] + [0] * (len(sizes) - 1)),
            )

            # Style pie chart text
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(11)

            format_axis_labels(ax5, "All Generated Driver Outcomes")
        else:
            ax5.text(
                0.5,
                0.5,
                "No driver\ndata available",
                ha="center",
                va="center",
                transform=ax5.transAxes,
                fontsize=12,
                style="italic",
                color=self.theme_colors["text_color"],
            )
            format_axis_labels(ax5, "All Generated Driver Outcomes")

        # 6. Algorithm complexity (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")

        complexity_text = "Algorithm Complexity Analysis\n"
        complexity_text += "═" * 70 + "\n"
        complexity_text += (
            "Routing (Shortest Paths):           O((V + E) log V) - V=nodes, E=edges\n"
        )
        complexity_text += "Pricing (Game Theory):  O(z²) - z=zones\n"
        complexity_text += (
            "Prediction (DP):        O(t x s² x w) - t=time, s=states, w=weather\n"
        )
        complexity_text += "Coordination (D&C):     O(z²/d + d²) - d=districts\n"
        complexity_text += "Overall System:         O(D x V log V + z²) - D=drivers"

        ax6.text(
            0.5,
            0.5,
            complexity_text,
            transform=ax6.transAxes,
            fontsize=12,
            ha="center",
            va="center",
            fontfamily="monospace",
            color=self.theme_colors["text_color"],
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": self.theme_colors["grid_color"],
                "alpha": 0.9,
                "edgecolor": self.theme_colors["main_color"],
                "linewidth": 2,
            },
        )

        plt.tight_layout()
        save_plot(fig, f"{self.output_dir}/summary_dashboard.png")

        print(f"Created summary dashboard: {self.output_dir}/summary_dashboard.png")

    def _create_static_complexity_chart(self):
        """Create a static complexity comparison chart with known algorithmic complexities"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define known algorithm complexities
        algorithms = [
            "Route Optimization\n(Shortest Paths)",
            "Dynamic Pricing\n(Game Theory)",
            "Demand Prediction\n(Dynamic Programming)",
            "City Coordination\n(Divide & Conquer)",
            "Zone Selection\n(Greedy Heuristic)",
        ]

        complexities = [
            "O((V + E) log V)",
            "O(z²)",
            "O(t x s² x w)",
            "O(z²/d + d²)",
            "O(z log z)",
        ]

        # Create bar positions
        y_pos = np.arange(len(algorithms))

        # Create horizontal bar chart
        bars = ax.barh(y_pos, np.arange(len(algorithms)), alpha=0.8)

        # Color code by algorithm type using theme colors
        for i, bar in enumerate(bars):
            bar.set_color(
                self.theme_colors["bar_colors"][
                    i % len(self.theme_colors["bar_colors"])
                ]
            )

        # Add complexity annotations
        for i, comp in enumerate(complexities):
            ax.text(
                0.1,
                i,
                comp,
                va="center",
                fontweight="bold",
                color=self.theme_colors["text_color"],
                fontsize=11,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(algorithms, fontsize=10)
        format_axis_labels(
            ax, "Algorithm Complexity Comparison", "Algorithmic Complexity"
        )
        ax.set_xlim(0, len(algorithms))

        # Remove x-axis as it's not meaningful
        ax.set_xticks([])

        # Add legend using theme colors
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][0],
                label="Routing (Shortest Paths)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][1],
                label="Pricing (Game Theory)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][2],
                label="Prediction (DP)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][3],
                label="Coordination (D&C)",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=self.theme_colors["bar_colors"][4],
                label="Selection (Greedy)",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        plt.tight_layout()
        save_plot(fig, f"{self.output_dir}/algorithm_comparison.png")


if __name__ == "__main__":
    # Test visualization with sample results
    visualizer = ParkingVisualizer("output/results.json")
    visualizer.create_all_visualizations()
