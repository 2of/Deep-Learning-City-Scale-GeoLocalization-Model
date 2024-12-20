import osmnx as ox
import geopandas as gpd

# Set the place name (e.g., "Chicago")
place_name = "Chicago, Illinois, USA"

# Download the road network for this place
graph = ox.graph_from_place(place_name, network_type='all')
# Convert the graph to a GeoDataFrame
nodes, edges = ox.graph_to_gdfs(graph)

# Save road network data to a shapefile for later use (optional)
edges.to_file("chicago_.shp")
