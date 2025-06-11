#!/usr/bin/env python3
"""
Calculate map tiles needed for Grand Rapids real-world data visualization
"""

import math
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.map_data_loader import get_map_data_loader


def deg2num(lat_deg, lon_deg, zoom):
    """
    Convert lat/lon coordinates to tile numbers at given zoom level.
    Based on OpenStreetMap tile numbering system.
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def calculate_tile_coverage(min_lat, min_lon, max_lat, max_lon, zoom_levels):
    """
    Calculate how many tiles are needed to cover a bounding box at different zoom levels.
    """
    results = []

    for zoom in zoom_levels:
        # Get tile coordinates for corners
        min_x, max_y = deg2num(min_lat, min_lon, zoom)  # Bottom-left
        max_x, min_y = deg2num(max_lat, max_lon, zoom)  # Top-right

        # Calculate tile counts
        tiles_x = max_x - min_x + 1
        tiles_y = max_y - min_y + 1
        total_tiles = tiles_x * tiles_y

        # Calculate area per tile at this zoom level
        world_tiles_at_zoom = 2**zoom
        (360.0 / world_tiles_at_zoom) * (180.0 / world_tiles_at_zoom)

        # Estimate tile size in meters (approximate, varies by latitude)
        earth_circumference = 40075017  # meters at equator
        meters_per_degree_lon = earth_circumference / 360.0
        meters_per_degree_lat = 111320  # approximately constant

        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon_at_lat = meters_per_degree_lon * math.cos(
            math.radians(avg_lat)
        )

        tile_width_meters = (360.0 / world_tiles_at_zoom) * meters_per_degree_lon_at_lat
        tile_height_meters = (180.0 / world_tiles_at_zoom) * meters_per_degree_lat

        results.append(
            {
                "zoom": zoom,
                "tiles_x": tiles_x,
                "tiles_y": tiles_y,
                "total_tiles": total_tiles,
                "tile_width_meters": tile_width_meters,
                "tile_height_meters": tile_height_meters,
                "area_per_tile_km2": (tile_width_meters * tile_height_meters) / 1e6,
            }
        )

    return results


def estimate_data_sizes(tile_counts):
    """
    Estimate data sizes for different tile types.
    """
    # Typical tile sizes (approximate)
    tile_sizes = {
        "raster_png": 15,  # KB per tile (street map)
        "raster_satellite": 25,  # KB per tile (satellite imagery)
        "vector_mvt": 8,  # KB per tile (Mapbox Vector Tiles)
        "vector_geojson": 12,  # KB per tile (GeoJSON equivalent)
    }

    results = {}
    for tile_type, size_kb in tile_sizes.items():
        results[tile_type] = []
        for data in tile_counts:
            total_size_kb = data["total_tiles"] * size_kb
            total_size_mb = total_size_kb / 1024
            results[tile_type].append(
                {
                    "zoom": data["zoom"],
                    "total_tiles": data["total_tiles"],
                    "size_kb": total_size_kb,
                    "size_mb": total_size_mb,
                    "size_gb": total_size_mb / 1024,
                }
            )

    return results


def main():
    """Calculate and display tile requirements for Grand Rapids data."""
    print("🗺️  Map Tile Calculator for Grand Rapids Real-World Data")
    print("=" * 70)

    try:
        # Get bounds from our real data
        map_loader = get_map_data_loader()
        if map_loader.is_data_available():
            bounds = map_loader.get_simulation_bounds()
            min_lat, min_lon, max_lat, max_lon = bounds
            print("📍 Coverage Area: Downtown Grand Rapids")
            print(
                f"   Latitude:  {min_lat:.6f}° to {max_lat:.6f}° ({max_lat - min_lat:.6f}° range)"
            )
            print(
                f"   Longitude: {min_lon:.6f}° to {max_lon:.6f}° ({max_lon - min_lon:.6f}° range)"
            )
        else:
            # Fallback to approximate downtown Grand Rapids bounds
            min_lat, min_lon = 42.956, -85.683
            max_lat, max_lon = 42.973, -85.668
            print("⚠️  Using approximate bounds (real data not available)")
            print("📍 Approximate Coverage: Downtown Grand Rapids")
            print(f"   Latitude:  {min_lat:.6f}° to {max_lat:.6f}°")
            print(f"   Longitude: {min_lon:.6f}° to {max_lon:.6f}°")

        # Calculate coverage area
        lat_km = (max_lat - min_lat) * 111.32  # 1 degree lat ≈ 111.32 km
        avg_lat = (min_lat + max_lat) / 2
        lon_km = (max_lon - min_lon) * 111.32 * math.cos(math.radians(avg_lat))
        area_km2 = lat_km * lon_km

        print(f"   Dimensions: {lat_km:.2f} km x {lon_km:.2f} km")
        print(f"   Total Area: {area_km2:.2f} km²")

    except Exception as e:
        print(f"Error loading map data: {e}")
        return

    # Zoom levels to analyze
    zoom_levels = [10, 12, 14, 15, 16, 17, 18, 19]

    print("\n📊 Tile Requirements Analysis")
    print("=" * 70)

    # Calculate tile coverage
    tile_data = calculate_tile_coverage(min_lat, min_lon, max_lat, max_lon, zoom_levels)

    # Display tile counts
    print(
        f"{'Zoom':<5} {'Tiles X':<8} {'Tiles Y':<8} {'Total':<8} {'Tile Size':<15} {'Area/Tile':<12}"
    )
    print("-" * 70)

    for data in tile_data:
        print(
            f"{data['zoom']:<5} {data['tiles_x']:<8} {data['tiles_y']:<8} {data['total_tiles']:<8} "
            f"{data['tile_width_meters']:.0f}x{data['tile_height_meters']:.0f}m{'':<5} "
            f"{data['area_per_tile_km2']:.3f} km²"
        )

    # Estimate data sizes
    print("\n💾 Estimated Storage Requirements")
    print("=" * 70)

    size_estimates = estimate_data_sizes(tile_data)

    for tile_type, estimates in size_estimates.items():
        print(f"\n{tile_type.replace('_', ' ').title()}:")
        print(
            f"{'Zoom':<5} {'Tiles':<8} {'Size (KB)':<12} {'Size (MB)':<12} {'Size (GB)':<10}"
        )
        print("-" * 50)

        for est in estimates:
            if est["size_gb"] < 0.001:
                size_str = f"{est['size_mb']:.1f} MB"
            else:
                size_str = f"{est['size_gb']:.2f} GB"

            print(
                f"{est['zoom']:<5} {est['total_tiles']:<8} {est['size_kb']:<12,.0f} "
                f"{est['size_mb']:<12,.1f} {size_str:<10}"
            )

    # Recommendations
    print("\n🎯 Recommendations")
    print("=" * 70)

    print("For Web Visualization:")
    print("  • Zoom 10-15: Good for overview and city-level planning")
    print("  • Zoom 16-18: Ideal for street-level parking analysis")
    print("  • Zoom 19+: Detailed spot-level visualization (high data cost)")

    print("\nFor Interactive Applications:")
    print("  • Cache tiles for zoom levels 12-17 for best performance")
    print("  • Use vector tiles (MVT) for smaller file sizes and styling flexibility")
    print("  • Consider on-demand loading for zoom levels 18+")

    print("\nFor Real-time Systems:")
    print("  • Zoom 14-16 provides good balance of detail vs. performance")
    print("  • Pre-generate tiles for frequently accessed areas")
    print("  • Use CDN for tile delivery in production")


if __name__ == "__main__":
    main()
