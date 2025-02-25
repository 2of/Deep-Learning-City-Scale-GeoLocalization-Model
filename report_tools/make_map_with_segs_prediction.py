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

# List of CSV file paths
file_paths = [
    "./predgroups/group_0.3786571_0.65925187.csv",

    "./predgroups/group_0.38926515_0.83749783.csv",
    "./predgroups/group_0.3735046_0.36372516.csv"
    
    # "./predgroups/group_0.38926515_0.83749783.csv"
    # Add more file paths as needed
]
file_paths = [
    "predgroups/group_0.395731_0.838402.csv",
    "predgroups/group_0.409572_0.8452477.csv",
    "predgroups/group_0.409572_0.8529975.csv",
    "predgroups/group_0.409572_0.85235167.csv",
    "predgroups/group_0.3682511_0.140401.csv",
    "predgroups/group_0.3682511_0.13962603.csv",
    "predgroups/group_0.3689583_0.1596464.csv",
    "predgroups/group_0.3689583_0.1604214.csv",
    "predgroups/group_0.3689583_0.16210052.csv",
    "predgroups/group_0.3703727_0.21195774.csv",
    "predgroups/group_0.3717871_0.2651732.csv",
    "predgroups/group_0.3724943_0.30314735.csv",
    "predgroups/group_0.3735046_0.36372516.csv",
    "predgroups/group_0.3786571_0.6433647.csv",
    "predgroups/group_0.3786571_0.65860605.csv"
]

# Colors for different groups
colors = ["orange", "red", "purple", "blue", "pink", "gray"]

# Process each CSV file
for i, file_path in enumerate(file_paths):
    # Load the CSV file, skipping the first row (if it contains headers)
    df = pd.read_csv(file_path, skiprows=1, header=None, names=["pred_lat", "pred_long", "actual_lat", "actual_long"])
    
    # Convert all columns to float
    df = df.astype(float)
    
    # Randomly sample 100 rows (or use the entire dataset)
    sample = df  # or sample = df.sample(n=100, random_state=4)
    
    # Lists to store denormalized predictions
    pred_lat_list, pred_long_list = [], []
    
    # Process each row and denormalize before plotting
    for _, row in sample.iterrows():
        pred_lat, pred_long = denormalize_coordinates(row["pred_lat"], row["pred_long"], bounding_box)
        actual_lat, actual_long = denormalize_coordinates(row["actual_lat"], row["actual_long"], bounding_box)
        
        pred_lat_list.append(pred_lat)
        pred_long_list.append(pred_long)
        
        # Add predicted location marker (color depends on the group)
        folium.Marker(
            location=[pred_lat, pred_long],
            popup=f"Prediction Group {i+1}",
            icon=folium.Icon(color=colors[i % len(colors)])
        ).add_to(m)
        
        # Add actual location marker (green)
        folium.Marker(
            location=[actual_lat, actual_long],
            popup=f"Actual Group {i+1}",
            icon=folium.Icon(color="green")
        ).add_to(m)
    
    # Calculate centroid of predictions for this group
    centroid_lat = sum(pred_lat_list) / len(pred_lat_list)
    centroid_long = sum(pred_long_list) / len(pred_long_list)
    
    # Add centroid marker (blue)
    folium.Marker(
        location=[centroid_lat, centroid_long],
        popup=f"Centroid of Predictions Group {i+1}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    # Draw lines between each prediction and the centroid
    for pred_lat, pred_long in zip(pred_lat_list, pred_long_list):
        folium.PolyLine(
            locations=[[pred_lat, pred_long], [centroid_lat, centroid_long]],
            color=colors[i % len(colors)],
            weight=4,
            opacity=0.6,
            popup=f"Prediction to Centroid Group {i+1}"
        ).add_to(m)
    
    # Draw a thicker line between the centroid and the true location
    folium.PolyLine(
        locations=[[centroid_lat, centroid_long], [actual_lat, actual_long]],
        color="black",
        weight=8,
        opacity=0.8,
        popup=f"Centroid to True Group {i+1}"
    ).add_to(m)

# Save to an HTML file
m.save("map.html")