import osmnx as ox
import geopandas as gpd
import os

def download_map(place_name, save_path='data/shapefiles'):
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Download the road network for the specified place
    graph = ox.graph_from_place(place_name, network_type='all')

    # Convert the graph to a GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(graph)

    # Save road network data to a shapefile
    shapefile_path = os.path.join(save_path, f"{place_name.replace(',', '').replace(' ', '_')}_roads.shp")
    edges.to_file(shapefile_path)

    print(f"Road network shapefile saved to {shapefile_path}")

# Example function call for testing purposes
# download_road_network("Chicago, Illinois, USA")