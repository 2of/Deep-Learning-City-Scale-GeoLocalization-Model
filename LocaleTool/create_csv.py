import argparse
import math
import pandas as pd
from tqdm import tqdm  # For the progress bar

# Constants
earth_radius_km = 6371  # Radius of the Earth in kilometers

def generate_points(center_lat, center_lon, radius_km, d, output_file):
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

    # Initialize progress bar for iteration
    total_steps = int((lat_end - lat_start) / d_lat)
    with tqdm(total=total_steps, desc="Processing Points", unit="step") as pbar:
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
            pbar.update(1)  # Update progress bar for each latitude step

    # Create a DataFrame from the points and save to CSV
    df = pd.DataFrame(points, columns=['Latitude', 'Longitude'])
    df.to_csv(output_file, index=False)
    print(f"Coordinates saved to {output_file}")


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Generate grid points within a circle.")
    parser.add_argument("lat", type=float, help="Center latitude")
    parser.add_argument("lon", type=float, help="Center longitude")
    parser.add_argument("radius", type=float, help="Radius of the circle in kilometers")
    parser.add_argument("distance", type=float, help="Distance between query points in meters")
    parser.add_argument("output_file", type=str, help="Output CSV file name")
    
    args = parser.parse_args()

    # Generate points and save to CSV
    generate_points(args.lat, args.lon, args.radius, args.distance, args.output_file)


if __name__ == "__main__":
    main()
