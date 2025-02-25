import pandas as pd
import folium
from folium.plugins import HeatMap
import branca

# Path to your CSV file
file_path = 'OVERALL_ATTN_WITH_STATS/preds_test_vs_actual.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

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

# Denormalize the coordinates
denormalized_data = []
for _, row in df.iterrows():
    pred_lat, pred_lon, true_lat, true_lon = row
    pred_lat_denorm, pred_lon_denorm = denormalize_coordinates(pred_lat, pred_lon, bounding_box)
    true_lat_denorm, true_lon_denorm = denormalize_coordinates(true_lat, true_lon, bounding_box)
    denormalized_data.append((pred_lat_denorm, pred_lon_denorm, true_lat_denorm, true_lon_denorm))

# Prepare data for the heatmap
heatmap_data = []
for row in denormalized_data:
    heatmap_data.append([row[0], row[1]])  # Predicted locations
    # heatmap_data.append([row[2], row[3]])  # True locations

# Calculate the center of the map
avg_lat = sum([row[0] for row in denormalized_data] + [row[2] for row in denormalized_data]) / (2 * len(denormalized_data))
avg_lon = sum([row[1] for row in denormalized_data] + [row[3] for row in denormalized_data]) / (2 * len(denormalized_data))

# Create the map
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=15)

# Add the heatmap with adjusted transparency, blur, and radius
HeatMap(
    heatmap_data, 
    radius=12,          
    blur=15,            
    min_opacity=0.3,    
    max_opacity=0.7     
).add_to(m)

# Create a color scale
colormap = branca.colormap.LinearColormap(
    colors=['blue', 'green', 'yellow', 'red'],  # Gradient from low to high density
    vmin=0,  # Min density (adjust if needed)
    vmax=1,  # Max density (adjust if needed)
    caption="Density of Predictions and True Locations"  # Legend label
)

# Add the scale to the map
colormap.add_to(m)

# Define bounding box coordinates
bounding_box_coords = [
    [bounding_box["min_lat"], bounding_box["min_lon"]],
    [bounding_box["min_lat"], bounding_box["max_lon"]],
    [bounding_box["max_lat"], bounding_box["max_lon"]],
    [bounding_box["max_lat"], bounding_box["min_lon"]],
    [bounding_box["min_lat"], bounding_box["min_lon"]]  # Close the polygon
]

# Draw the bounding box as a semi-transparent polygon
folium.Polygon(
    locations=bounding_box_coords,
    color="green",          # Outline color
    weight=3,               # Outline thickness
    fill=True,              # Enable fill
    fill_color="lightgreen", # Fill color
    fill_opacity=0.2,       # Slightly transparent fill
    popup="Bounding Box"
).add_to(m)

# Save the map
m.save('heatmap.html')

# Display the map in Jupyter Notebook (if using one)
m