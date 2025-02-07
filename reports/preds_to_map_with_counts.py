import pandas as pd
import folium
from folium.features import DivIcon

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

# Load the CSV file
file_path = "predictions.csv"
df = pd.read_csv(file_path)

# Convert all columns to float (except '# of detections' which is an integer)
df = df.astype({
    "Predicted Latitude": float,
    "Predicted Longitude": float,
    "True Latitude": float,
    "True Longitude": float,
    "# of detections": int
})

# Randomly sample 100 rows
sample = df.sample(n=35, random_state=4)

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

# Process each row and denormalize before plotting
for _, row in sample.iterrows():
    pred_lat, pred_long = denormalize_coordinates(row["Predicted Latitude"], row["Predicted Longitude"], bounding_box)
    actual_lat, actual_long = denormalize_coordinates(row["True Latitude"], row["True Longitude"], bounding_box)
    num_detections = row["# of detections"]

    # Add predicted location marker (orange)
    folium.Marker(
        location=[pred_lat, pred_long],
        popup=f"Prediction<br>Detections: {num_detections}",
        icon=folium.Icon(color="orange")
    ).add_to(m)

    # Add actual location marker (green)
    folium.Marker(
        location=[actual_lat, actual_long],
        popup=f"Actual<br>Detections: {num_detections}",
        icon=folium.Icon(color="green")
    ).add_to(m)

    # Add text labels for the number of detections on the pins
    folium.map.Marker(
        [pred_lat, pred_long],
        icon=DivIcon(
            icon_size=(50, 50),
            icon_anchor=(12, 12),
            html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">{num_detections}</div>'
        )
    ).add_to(m)

    folium.map.Marker(
        [actual_lat, actual_long],
        icon=DivIcon(
            icon_size=(50, 50),
            icon_anchor=(12, 12),
            html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">{num_detections}</div>'
        )
    ).add_to(m)

    # Draw a line between the prediction and actual points (optional)
    folium.PolyLine(
        locations=[[pred_lat, pred_long], [actual_lat, actual_long]],
        color="magenta",
        weight=2,
        opacity=0.7
    ).add_to(m)

# Save to an HTML file and display
m.save("map_with_detections.html")