import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
import numpy as np

# Load the road network shapefile into a GeoDataFrame
gdf = gpd.read_file("chicago_.shp")

# List to store the generated points
points = []

# Desired distance between points (in meters)
distance = 1

# Function to get evenly spaced points along a LineString
def get_points_on_line(line, distance):
    # Create a list of points along the line at specified intervals
    num_points = int(line.length // distance)  # How many points to sample
    points = []
    
    # Iterate over the length of the line and sample points
    for i in range(num_points + 1):
        point = line.interpolate(i * distance)  # Get point at distance i
        points.append((point.y, point.x))  # Store lat, lon (y = lat, x = lon)
    
    return points

# Iterate through each road in the GeoDataFrame
for _, row in gdf.iterrows():
    # Each row is a road (LineString geometry)
    line = row['geometry']
    
    if isinstance(line, LineString):  # Ensure it's a LineString geometry
        road_points = get_points_on_line(line, distance)
        points.extend(road_points)

# Create a DataFrame to store the points
points_df = pd.DataFrame(points, columns=['Latitude', 'Longitude'])

# Save the points to a CSV file
points_df.to_csv("chicago_road_points.csv", index=False)

print(f"Points saved to chicago_road_points.csv")
