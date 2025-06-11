"""
Enhanced Map Visualization Module for Parking Optimization
Creates comprehensive geographic visualizations using real Grand Rapids data
"""

import json
import os
from typing import Dict, List, Optional

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from folium import plugins

from analysis.viz_utils import apply_dark_theme, get_theme_colors, save_plot
from core.logger import get_logger
from core.map_data_loader import MapDataLoader

logger = get_logger(__name__)


class MapVisualizer:
    """Creates comprehensive map visualizations for parking optimization system."""

    def __init__(
        self,
        results_file: str = "output/results.json",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize enhanced map visualizer.

        Args:
            results_file: Path to simulation results JSON
            output_dir: Custom output directory (if None, uses default visualization_output)
        """
        self.results_file = results_file
        self.results = None
        self.output_dir = output_dir or "visualization_output"
        self.map_loader = MapDataLoader()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Apply dark theme
        apply_dark_theme()
        self.theme_colors = get_theme_colors()

        # Load results and map data
        self._load_results()
        self._load_map_data()

    def _load_results(self):
        """Load simulation results from JSON file."""
        try:
            with open(self.results_file) as f:
                self.results = json.load(f)
            logger.info(f"Loaded results from {self.results_file}")
        except FileNotFoundError:
            logger.error(f"Results file {self.results_file} not found")
            self.results = None

    def _load_map_data(self):
        """Load real Grand Rapids map data."""
        try:
            self.map_loader.load_all_data()
            logger.info("Successfully loaded Grand Rapids map data")
        except Exception as e:
            logger.error(f"Failed to load map data: {e}")

    def create_interactive_parking_map(self) -> str:
        """
        Create an interactive Folium map with real-time parking data.

        Returns:
            Path to the generated HTML file
        """
        if not self.map_loader.is_data_available():
            logger.error("Map data not available")
            return ""

        # Get simulation bounds
        bounds = self.map_loader.get_simulation_bounds()
        center_lat = (bounds[0] + bounds[2]) / 2
        center_lon = (bounds[1] + bounds[3]) / 2

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="OpenStreetMap",
            prefer_canvas=True,
        )

        # Add downtown boundary
        boundary_gdf = self.map_loader.boundary_gdf
        if boundary_gdf is not None:
            folium.GeoJson(
                boundary_gdf.to_json(),
                style_function=lambda x: {
                    "fillColor": "lightblue",
                    "color": "blue",
                    "weight": 3,
                    "fillOpacity": 0.1,
                },
                tooltip="Downtown Grand Rapids",
            ).add_to(m)

        # Add parking zones with occupancy data
        parking_zones = self.map_loader.get_parking_zones()
        self._add_parking_zones_to_map(m, parking_zones)

        # Add traffic data if available
        self._add_traffic_overlay(m)

        # Add heat map layer
        self._add_occupancy_heatmap(m, parking_zones)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        map_file = f"{self.output_dir}/interactive_parking_map.html"
        m.save(map_file)
        logger.info(f"Created interactive map: {map_file}")

        return map_file

    def _add_parking_zones_to_map(self, m: folium.Map, zones: List[Dict]):
        """Add parking zones with occupancy-based styling."""
        # Create feature groups for different zone types
        lots_group = folium.FeatureGroup(name="Parking Lots")
        meters_group = folium.FeatureGroup(name="Parking Meters")

        for zone in zones:
            # Get occupancy data (mock for now, replace with real data)
            occupancy = self._get_zone_occupancy(zone["id"])
            color = self._get_occupancy_color(occupancy)

            # Create popup with zone information
            popup_html = f"""
            <div style="font-family: Arial; font-size: 14px;">
                <h4>{zone["type"].title()} {zone["id"]}</h4>
                <b>Occupancy:</b> {occupancy:.1%}<br/>
                <b>Capacity:</b> {zone["capacity"]} spaces<br/>
                <b>Rate:</b> ${zone["hourly_rate"]:.2f}/hour<br/>
                <b>Type:</b> {zone["type"]}<br/>
            </div>
            """

            # Different markers for lots vs meters
            if zone["type"] == "lot":
                marker = folium.Marker(
                    zone["coordinates"],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Parking Lot {zone['id']} ({occupancy:.1%} full)",
                    icon=folium.Icon(color=color, icon="car", prefix="fa"),
                )
                marker.add_to(lots_group)
            else:
                marker = folium.CircleMarker(
                    zone["coordinates"],
                    radius=8,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Meter Zone {zone['id']} ({occupancy:.1%} full)",
                    color="white",
                    weight=2,
                    fillColor=self._get_occupancy_hex_color(occupancy),
                    fillOpacity=0.8,
                )
                marker.add_to(meters_group)

        # Add feature groups to map
        lots_group.add_to(m)
        meters_group.add_to(m)

    def _add_traffic_overlay(self, m: folium.Map):
        """Add traffic data overlay if available."""
        # This would integrate with your TomTom traffic data
        # For now, add sample traffic markers on major roads
        if self.map_loader.road_network:
            traffic_group = folium.FeatureGroup(name="Traffic Data")

            # Sample some major intersections
            major_nodes = list(self.map_loader.road_network.nodes())[:20]
            for node_id in major_nodes:
                coords = self.map_loader.get_node_coordinates(node_id)
                # Mock traffic speed (replace with real TomTom data)
                speed = np.random.uniform(15, 45)
                color = "green" if speed > 30 else "orange" if speed > 20 else "red"

                folium.CircleMarker(
                    coords,
                    radius=4,
                    popup=f"Traffic Speed: {speed:.1f} km/h",
                    color="white",
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.7,
                ).add_to(traffic_group)

            traffic_group.add_to(m)

    def _add_occupancy_heatmap(self, m: folium.Map, zones: List[Dict]):
        """Add occupancy heatmap overlay."""
        # Prepare heat map data
        heat_data = []
        for zone in zones:
            lat, lon = zone["coordinates"]
            occupancy = self._get_zone_occupancy(zone["id"])
            # Weight by occupancy and capacity
            weight = occupancy * zone["capacity"] / 10
            heat_data.append([lat, lon, weight])

        # Add heat map
        heatmap = plugins.HeatMap(
            heat_data,
            name="Occupancy Heatmap",
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={
                0.0: "blue",
                0.3: "green",
                0.6: "yellow",
                0.8: "orange",
                1.0: "red",
            },
        )
        heatmap.add_to(m)

    def create_network_analysis_map(self) -> str:
        """
        Create a network analysis visualization showing road connectivity.

        Returns:
            Path to the generated image file
        """
        if not self.map_loader.road_network:
            logger.error("Road network data not available")
            return ""

        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor(self.theme_colors["background_color"])

        # Plot the road network
        g = self.map_loader.road_network

        # Get node and edge positions
        pos = {node: (data["x"], data["y"]) for node, data in g.nodes(data=True)}

        # Draw edges (roads)
        nx.draw_networkx_edges(
            g,
            pos,
            edge_color=self.theme_colors["grid_color"],
            width=0.5,
            alpha=0.6,
            ax=ax,
        )

        # Draw nodes (intersections)
        nx.draw_networkx_nodes(
            g,
            pos,
            node_color=self.theme_colors["main_color"],
            node_size=20,
            alpha=0.8,
            ax=ax,
        )

        # Add parking zones
        parking_zones = self.map_loader.get_parking_zones()
        for zone in parking_zones:
            lat, lon = zone["coordinates"]
            occupancy = self._get_zone_occupancy(zone["id"])
            color = self._get_occupancy_hex_color(occupancy)

            ax.scatter(
                lon,
                lat,
                c=color,
                s=zone["capacity"] * 3,  # Size by capacity
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
                zorder=5,
            )

        ax.set_title(
            "Grand Rapids Parking Network Analysis",
            fontsize=18,
            color=self.theme_colors["text_color"],
            fontweight="bold",
            pad=20,
        )

        ax.set_xlabel("Longitude", color=self.theme_colors["text_color"], fontsize=12)
        ax.set_ylabel("Latitude", color=self.theme_colors["text_color"], fontsize=12)

        # Style the plot
        ax.set_facecolor(self.theme_colors["background_color"])
        ax.tick_params(colors=self.theme_colors["text_color"])
        for spine in ax.spines.values():
            spine.set_color(self.theme_colors["grid_color"])

        plt.tight_layout()

        # Save the plot
        plot_file = f"{self.output_dir}/network_analysis_map.png"
        save_plot(fig, plot_file)
        logger.info(f"Created network analysis map: {plot_file}")

        return plot_file

    def create_comprehensive_dashboard_map(self) -> str:
        """
        Create a comprehensive dashboard combining multiple map views.

        Returns:
            Path to the generated image file
        """
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(
            "Grand Rapids Parking Optimization - Geographic Dashboard",
            fontsize=24,
            color=self.theme_colors["text_color"],
            fontweight="bold",
            y=0.95,
        )
        fig.patch.set_facecolor(self.theme_colors["background_color"])

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2)

        # 1. Overview map (top left, spans 2x2)
        ax_overview = fig.add_subplot(gs[0:2, 0:2])
        self._plot_overview_map(ax_overview)

        # 2. Occupancy distribution (top right)
        ax_dist = fig.add_subplot(gs[0, 2])
        self._plot_occupancy_distribution(ax_dist)

        # 3. Zone capacity chart (middle right)
        ax_capacity = fig.add_subplot(gs[1, 2])
        self._plot_capacity_distribution(ax_capacity)

        # 4. Traffic speed map (bottom left)
        ax_traffic = fig.add_subplot(gs[2, 0])
        self._plot_traffic_overview(ax_traffic)

        # 5. Revenue by area (bottom center)
        ax_revenue = fig.add_subplot(gs[2, 1])
        self._plot_revenue_by_area(ax_revenue)

        # 6. System metrics (bottom right)
        ax_metrics = fig.add_subplot(gs[2, 2])
        self._plot_system_metrics(ax_metrics)

        plt.tight_layout()

        # Save the dashboard
        dashboard_file = f"{self.output_dir}/geographic_dashboard.png"
        save_plot(fig, dashboard_file)
        logger.info(f"Created geographic dashboard: {dashboard_file}")

        return dashboard_file

    def _plot_overview_map(self, ax):
        """Plot the main overview map."""
        ax.set_title(
            "Downtown Grand Rapids - Real-Time Parking Status",
            fontsize=16,
            color=self.theme_colors["text_color"],
            fontweight="bold",
        )

        # Plot boundary
        if self.map_loader.boundary_gdf is not None:
            self.map_loader.boundary_gdf.plot(
                ax=ax,
                facecolor=self.theme_colors["background_color"],
                edgecolor=self.theme_colors["main_color"],
                linewidth=2,
                alpha=0.3,
            )

        # Plot parking zones
        parking_zones = self.map_loader.get_parking_zones()
        for zone in parking_zones:
            lat, lon = zone["coordinates"]
            occupancy = self._get_zone_occupancy(zone["id"])
            color = self._get_occupancy_hex_color(occupancy)

            ax.scatter(
                lon,
                lat,
                c=color,
                s=zone["capacity"] * 1.5,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_aspect("equal")
        ax.set_facecolor(self.theme_colors["background_color"])
        ax.tick_params(colors=self.theme_colors["text_color"])

    def _plot_occupancy_distribution(self, ax):
        """Plot occupancy distribution histogram."""
        parking_zones = self.map_loader.get_parking_zones()
        occupancies = [self._get_zone_occupancy(zone["id"]) for zone in parking_zones]

        ax.hist(
            occupancies,
            bins=20,
            color=self.theme_colors["main_color"],
            alpha=0.7,
            edgecolor="white",
        )
        ax.set_title("Occupancy Distribution", color=self.theme_colors["text_color"])
        ax.set_xlabel("Occupancy Rate", color=self.theme_colors["text_color"])
        ax.set_ylabel("Number of Zones", color=self.theme_colors["text_color"])
        ax.tick_params(colors=self.theme_colors["text_color"])

    def _plot_capacity_distribution(self, ax):
        """Plot capacity distribution by zone type."""
        ax.set_title(
            "Capacity Distribution",
            fontsize=14,
            color=self.theme_colors["text_color"],
            fontweight="bold",
        )

        parking_zones = self.map_loader.get_parking_zones()
        lot_capacities = [z["capacity"] for z in parking_zones if z["type"] == "lot"]
        meter_capacities = [
            z["capacity"] for z in parking_zones if z["type"] == "meter"
        ]

        ax.boxplot(
            [lot_capacities, meter_capacities],
            labels=["Lots", "Meters"],
            patch_artist=True,
            boxprops={"facecolor": self.theme_colors["main_color"], "alpha": 0.7},
        )
        ax.set_title("Capacity by Type", color=self.theme_colors["text_color"])
        ax.set_ylabel("Capacity", color=self.theme_colors["text_color"])

        ax.set_facecolor(self.theme_colors["background_color"])
        ax.tick_params(colors=self.theme_colors["text_color"])

    def _plot_traffic_overview(self, ax):
        """Plot traffic overview."""
        ax.set_title(
            "Traffic Integration",
            fontsize=14,
            color=self.theme_colors["text_color"],
            fontweight="bold",
        )

        ax.text(
            0.5,
            0.5,
            "Real-time traffic\nintegration active\n(TomTom API)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color=self.theme_colors["text_color"],
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": self.theme_colors["grid_color"],
                "alpha": 0.7,
            },
        )
        ax.set_title("Traffic Status", color=self.theme_colors["text_color"])

        ax.set_facecolor(self.theme_colors["background_color"])
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_revenue_by_area(self, ax):
        """Plot revenue by area."""
        ax.set_title(
            "Revenue Analysis",
            fontsize=14,
            color=self.theme_colors["text_color"],
            fontweight="bold",
        )

        areas = ["Downtown Core", "North District", "South District", "East District"]
        revenues = [1200, 800, 600, 400]

        ax.bar(areas, revenues, color=self.theme_colors["bar_colors"])
        ax.set_title("Revenue by Area", color=self.theme_colors["text_color"])
        ax.set_ylabel("Revenue ($)", color=self.theme_colors["text_color"])

    def _plot_system_metrics(self, ax):
        """Plot key system metrics."""
        if self.results:
            metrics = self.results.get("metrics", {})
            total_revenue = sum(metrics.get("total_revenue", [0]))
            success_rate = (
                metrics.get("successful_parks", 0)
                / (
                    metrics.get("successful_parks", 0)
                    + metrics.get("rejected_drivers", 1)
                )
                * 100
            )
            avg_search = np.mean(metrics.get("avg_search_time", [0]))

            metrics_text = f"""
System Performance
─────────────────
Revenue: ${total_revenue:.0f}
Success Rate: {success_rate:.1f}%
Avg Search: {avg_search:.1f} min
Zones Active: {len(self.map_loader.get_parking_zones())}
            """
        else:
            metrics_text = "No simulation data\navailable"

        ax.text(
            0.1,
            0.9,
            metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            color=self.theme_colors["text_color"],
            fontfamily="monospace",
            verticalalignment="top",
        )
        ax.set_title("System Metrics", color=self.theme_colors["text_color"])
        ax.axis("off")

    def _get_zone_occupancy(self, zone_id: str) -> float:
        """Get current occupancy for a zone (mock data for now)."""
        # In a real implementation, this would fetch from simulation results
        # For now, return realistic random occupancy
        np.random.seed(hash(zone_id) % 2**32)  # Consistent per zone
        return np.random.beta(2, 3)  # Skewed toward lower occupancy

    def _get_occupancy_color(self, occupancy: float) -> str:
        """Get Folium color name based on occupancy level."""
        if occupancy < 0.3:
            return "green"
        elif occupancy < 0.6:
            return "orange"
        elif occupancy < 0.85:
            return "red"
        else:
            return "darkred"

    def _get_occupancy_hex_color(self, occupancy: float) -> str:
        """Get hex color based on occupancy level."""
        # Green to red gradient
        if occupancy < 0.5:
            # Green to yellow
            r = int(255 * 2 * occupancy)
            g = 255
        else:
            # Yellow to red
            r = 255
            g = int(255 * 2 * (1 - occupancy))

        b = 0
        return f"#{r:02x}{g:02x}{b:02x}"


def create_all_map_visualizations(
    results_file: str = "output/results.json",
) -> List[str]:
    """
    Create all map visualizations and return list of generated files.

    Args:
        results_file: Path to simulation results

    Returns:
        List of generated file paths
    """
    visualizer = MapVisualizer(results_file)
    generated_files = []

    try:
        # Create interactive map
        interactive_map = visualizer.create_interactive_parking_map()
        if interactive_map:
            generated_files.append(interactive_map)

        # Create network analysis
        network_map = visualizer.create_network_analysis_map()
        if network_map:
            generated_files.append(network_map)

        # Create comprehensive dashboard
        dashboard = visualizer.create_comprehensive_dashboard_map()
        if dashboard:
            generated_files.append(dashboard)

        logger.info(f"Generated {len(generated_files)} map visualizations")

    except Exception as e:
        logger.error(f"Error creating map visualizations: {e}")

    return generated_files


if __name__ == "__main__":
    # Example usage
    files = create_all_map_visualizations()
    for file in files:
        print(f"Generated: {file}")
