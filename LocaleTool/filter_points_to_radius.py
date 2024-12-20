import pandas as pd
import math
from geopy.distance import geodesic
import sys
from tqdm import tqdm  # For the progress bar

# Function to calculate the distance between two lat, lon points using geodesic
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

# Function to filter points within a radius of the center point
def filter_points(input_csv, output_csv, center_lat, center_lon, radius):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # List to store points within the radius
    filtered_points = []
    
    # Iterate through the DataFrame with progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Filtering Points"):
        lat = row['Latitude']
        lon = row['Longitude']
        
        # Calculate distance from the center point
        distance = calculate_distance(center_lat, center_lon, lat, lon)
        
        # If the point is within the radius, keep it
        if distance <= radius:
            filtered_points.append((lat, lon))
    
    # Create a new DataFrame with the filtered points
    filtered_df = pd.DataFrame(filtered_points, columns=['Latitude', 'Longitude'])
    
    # Save the filtered points to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered points saved to {output_csv}")

if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 6:
        print("Usage: python filter_points.py <input_csv> <output_csv> <center_lat> <center_lon> <radius>")
        sys.exit(1)

    # Parse command line arguments
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    center_lat = float(sys.argv[3])
    center_lon = float(sys.argv[4])
    radius = float(sys.argv[5])

    # Call the filter function
    filter_points(input_csv, output_csv, center_lat, center_lon, radius)
