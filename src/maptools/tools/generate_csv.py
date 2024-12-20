import math
import pandas as pd
import os

# Constants
earth_radius_km = 6371  # Radius of the Earth in kilometers

def generate_points(center_lat, center_lon, radius_km, d, output_file_prefix):
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

    # Initialize progress tracking
    total_steps = max(1, int((lat_end - lat_start) / d_lat))  # Ensure total_steps is at least 1 to avoid division by zero
    progress_increment = max(1, total_steps // 30)  # Ensure progress_increment is at least 1 to avoid division by zero
    progress_counter = 0

    file_counter = 1
    points_per_file = 2000

    # Create the output directory if it doesn't exist
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

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
            
            if len(points) >= points_per_file:
                df = pd.DataFrame(points, columns=['Latitude', 'Longitude'])
                df.to_csv(os.path.join(output_dir, f"{output_file_prefix}_{file_counter}.csv"), index=False)
                print(f"Coordinates saved to {output_dir}/{output_file_prefix}_{file_counter}.csv")
                points.clear()
                file_counter += 1
            
            lon += d_lon
        
        lat += d_lat
        progress_counter += 1
        if progress_counter % progress_increment == 0:
            print(f"{progress_counter // progress_increment}/10")

    # Save any remaining points to a new file
    if points:
        df = pd.DataFrame(points, columns=['Latitude', 'Longitude'])
        df.to_csv(os.path.join(output_dir, f"{output_file_prefix}_{file_counter}.csv"), index=False)
        print(f"Coordinates saved to {output_dir}/{output_file_prefix}_{file_counter}.csv")