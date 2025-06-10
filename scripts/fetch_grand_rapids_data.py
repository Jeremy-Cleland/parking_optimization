import os

import geopandas as gpd
import osmnx as ox
import requests

# --- Configuration ---
DATA_URLS = {
    "downtown_boundary": "https://raw.githubusercontent.com/downtowngr/maps/master/boards/dda_boundary.geojson",
    "car_parking": "https://raw.githubusercontent.com/downtowngr/maps/master/transportation/car_parking.geojson",
    "parking_meters": "https://raw.githubusercontent.com/downtowngr/maps/master/transportation/parking_meters.geojson",
}
OUTPUT_DIR = "output/map_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_geojson(url: str, filepath: str):
    """Downloads a GeoJSON file and saves it."""
    print(f"Downloading data from {url}...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(filepath, "w") as f:
            f.write(r.text)
        print(f"Successfully saved to {filepath}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        raise


def fetch_road_network(boundary_gdf: gpd.GeoDataFrame, filepath: str):
    """Fetches road network from OpenStreetMap for the given boundary."""
    print("Fetching road network from OpenStreetMap using OSMnx...")
    # Get the bounding box of the boundary polygon
    boundary_polygon = boundary_gdf.unary_union

    # Download the network
    # Use 'drive' network type for car navigation
    graph = ox.graph_from_polygon(boundary_polygon, network_type="drive")

    print("Saving road network graph to disk...")
    # Save the graph as a .graphml file
    ox.save_graphml(graph, filepath)
    print(f"Road network saved to {filepath}")


def main():
    """Main function to download and process all map data."""
    print("--- Starting Grand Rapids Map Data Acquisition ---")

    # --- Download GeoJSON data ---
    boundary_path = os.path.join(OUTPUT_DIR, "downtown_boundary.geojson")
    parking_lots_path = os.path.join(OUTPUT_DIR, "parking_lots.geojson")
    parking_meters_path = os.path.join(OUTPUT_DIR, "parking_meters.geojson")

    download_geojson(DATA_URLS["downtown_boundary"], boundary_path)
    download_geojson(DATA_URLS["car_parking"], parking_lots_path)
    download_geojson(DATA_URLS["parking_meters"], parking_meters_path)

    # --- Load boundary to fetch road network ---
    print("\n--- Processing Road Network ---")
    if os.path.exists(boundary_path):
        boundary_gdf = gpd.read_file(boundary_path)
        network_path = os.path.join(OUTPUT_DIR, "grand_rapids_drive_network.graphml")
        fetch_road_network(boundary_gdf, network_path)

    print("\n--- Data acquisition complete! ---")
    print(f"All data saved in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
