import os
import pandas as pd
import folium

# Function to denormalize coordinates
def denormalize_coordinates(lat, lon, bounding_box):
    lat = float(lat) * (bounding_box['max_lat'] - bounding_box['min_lat']) + bounding_box['min_lat']
    lon = float(lon) * (bounding_box['max_lon'] - bounding_box['min_lon']) + bounding_box['min_lon']
    return lat, lon

# Bounding box values
bounding_box = {
    'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332,
    'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706
}

# Calculate center of bounding box
map_center = [
    (bounding_box["max_lat"] + bounding_box["min_lat"]) / 2,
    (bounding_box["max_lon"] + bounding_box["min_lon"]) / 2
]

# Create a folium map centered on the bounding box midpoint
m = folium.Map(location=map_center, zoom_start=14)

# Define bounding box coordinates
bounding_box_coords = [
    [bounding_box["min_lat"], bounding_box["min_lon"]],
    [bounding_box["min_lat"], bounding_box["max_lon"]],
    [bounding_box["max_lat"], bounding_box["max_lon"]],
    [bounding_box["max_lat"], bounding_box["min_lon"]],
]

# Draw filled bounding box
folium.Polygon(
    locations=bounding_box_coords,
    color="green",          # Outline color
    weight=2,               # Outline thickness
    fill=True,              # Enable fill
    fill_color="lightgreen", # Fill color
    fill_opacity=0.3,       # Semi-transparent fill
    popup="Bounding Box"
).add_to(m)

# Directory containing CSV files
directory = "predgroups"

# Colors for prediction and actual locations
prediction_color = "orange"
actual_color = "green"

# Process each CSV file in the directory
for i, file_name in enumerate(os.listdir(directory)):
    file_path = os.path.join(directory, file_name)
    
    # Load the CSV file, skipping the first row (if it contains headers)
    df = pd.read_csv(file_path, skiprows=1, header=None, names=["pred_lat", "pred_long", "actual_lat", "actual_long"])
    
    # Convert all columns to float
    df = df.astype(float)
    
    # Lists to store denormalized predictions and actuals
    pred_lat_list, pred_long_list = [], []
    actual_lat_list, actual_long_list = [], []
    
    # Process each row and denormalize before plotting
    for _, row in df.iterrows():
        pred_lat, pred_long = denormalize_coordinates(row["pred_lat"], row["pred_long"], bounding_box)
        actual_lat, actual_long = denormalize_coordinates(row["actual_lat"], row["actual_long"], bounding_box)
        
        pred_lat_list.append(pred_lat)
        pred_long_list.append(pred_long)
        actual_lat_list.append(actual_lat)
        actual_long_list.append(actual_long)
    
    # Calculate centroid of predictions for this group
    centroid_pred_lat = sum(pred_lat_list) / len(pred_lat_list)
    centroid_pred_long = sum(pred_long_list) / len(pred_long_list)
    
    # Add centroid marker for predictions (orange)
    folium.Marker(
        location=[centroid_pred_lat, centroid_pred_long],
        popup=f"Centroid of Predictions Group {i+1}",
        icon=folium.Icon(color=prediction_color)
    ).add_to(m)
    
    # Add actual location marker (green)
    folium.Marker(
        location=[actual_lat_list[0], actual_long_list[0]],
        popup=f"Actual Location Group {i+1}",
        icon=folium.Icon(color=actual_color)
    ).add_to(m)

# Save to an HTML file
m.save("map.html")