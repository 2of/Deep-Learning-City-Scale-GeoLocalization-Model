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

# Load the CSV file, skipping the first row (if it contains headers)
file_path = "OVERALL_NO_ATTN_WITH_STATS/preds_test_vs_actual.csv"
df = pd.read_csv(file_path, skiprows=1, header=None, names=["pred_lat", "pred_long", "actual_lat", "actual_long"])

# Convert all columns to float
df = df.astype(float)

# Randomly sample 100 rows
sample = df.sample(n=100, random_state=4)

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
    pred_lat, pred_long = denormalize_coordinates(row["pred_lat"], row["pred_long"], bounding_box)
    actual_lat, actual_long = denormalize_coordinates(row["actual_lat"], row["actual_long"], bounding_box)

    # Add predicted location marker (blue)
    folium.Marker(
        location=[pred_lat, pred_long],
        popup="Prediction",
        icon=folium.Icon(color="orange")
    ).add_to(m)

    # Add actual location marker (red)
    folium.Marker(
        location=[actual_lat, actual_long],
        popup="Actual",
        icon=folium.Icon(color="green")
    ).add_to(m)

    # Draw a line between the prediction and actual points
    # folium.PolyLine(
    #     locations=[[pred_lat, pred_long], [actual_lat, actual_long]],
    #     color="magenta",
    #     weight=2,
    #     opacity=0.7
    # ).add_to(m)

# Save to an HTML file and display
m.save("map_preds.html")