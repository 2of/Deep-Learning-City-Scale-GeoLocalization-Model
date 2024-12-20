import math
import pandas as pd

# Constants
d = 20 # Distance between each query point in meters
earth_radius_km = 6371  # Radius of the Earth in kilometers

# Define the center point (example coordinates for New York City)
center_lat = 40.7128
center_lon = -74.0060

# Convert radius in meters to kilometers
radii = [1, 5, 10, 11, 12]  # Radii in kilometers
radius_km = 2  # Using 2 km for demonstration

# Calculate approximate degree change for each step in latitude/longitude
d_lat = d / (earth_radius_km * 1000) * (180 / math.pi)
d_lon = d / (earth_radius_km * 1000) * (180 / math.pi) / math.cos(math.radians(center_lat))

# Generate grid points within the circle
points = []

# Iterate over latitude and longitude steps within the square that bounds the circle
lat_start = center_lat - radius_km * (180 / math.pi) / earth_radius_km
lat_end = center_lat + radius_km * (180 / math.pi) / earth_radius_km
lon_start = center_lon - radius_km * (180 / math.pi) / (earth_radius_km * math.cos(math.radians(center_lat)))
lon_end = center_lon + radius_km * (180 / math.pi) / (earth_radius_km * math.cos(math.radians(center_lat)))

lat = lat_start
while lat <= lat_end:
    lon = lon_start
    while lon <= lon_end:
        # Calculate the distance from the center point to (lat, lon)
        distance = earth_radius_km * math.acos(
            math.sin(math.radians(center_lat)) * math.sin(math.radians(lat)) +
            math.cos(math.radians(center_lat)) * math.cos(math.radians(lat)) * math.cos(math.radians(center_lon - lon))
        )
        
        # Check if the point is within the radius
        if distance <= radius_km:
            points.append((lat, lon))
        lon += d_lon
    lat += d_lat

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(points, columns=['Latitude', 'Longitude'])
df.to_csv("coordinates.csv", index=False)

print("Coordinates saved to coordinates.csv")
