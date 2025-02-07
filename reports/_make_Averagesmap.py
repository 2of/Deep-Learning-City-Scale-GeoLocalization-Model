import pandas as pd
import folium
import math

# Load the averaged predictions CSV
file_path = 'averaged_predictions.csv'
df = pd.read_csv(file_path)

# Take only a random sample of 3 rows for visualization
df = df.sample(3)

# Function to denormalize coordinates
def denormalize_coordinates(lat, lon, bounding_box):
    lat = float(lat) * (bounding_box['max_lat'] - bounding_box['min_lat']) + bounding_box['min_lat']
    lon = float(lon) * (bounding_box['max_lon'] - bounding_box['min_lon']) + bounding_box['min_lon']
    return lat, lon

# Function to compute Euclidean distance (MSE proxy)
def euclidean_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

# Bounding box values
bounding_box = {
    'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332,
    'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706
}

# Denormalize all points in the dataframe
for col in ["True Latitude", "Averaged Pred Latitude (SEGS)", "Pred Latitude (Overall)", "Final Averaged Latitude"]:
    df[col], df[col.replace("Latitude", "Longitude")] = zip(*df.apply(
        lambda row: denormalize_coordinates(row[col], row[col.replace("Latitude", "Longitude")], bounding_box),
        axis=1
    ))

# Create a folium map centered on the average of True Latitude and True Longitude
map_center = [df["True Latitude"].mean(), df["True Longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=12)

# Process each row
for _, row in df.iterrows():
    true_lat, true_lon = row["True Latitude"], row["True Longitude"]
    segs_lat, segs_lon = row["Averaged Pred Latitude (SEGS)"], row["Averaged Pred Longitude (SEGS)"]
    overall_lat, overall_lon = row["Pred Latitude (Overall)"], row["Pred Longitude (Overall)"]
    final_avg_lat, final_avg_lon = row["Final Averaged Latitude"], row["Final Averaged Longitude"]

    # Calculate Euclidean distances (MSE approximation)
    mse_segs = euclidean_distance(true_lat, true_lon, segs_lat, segs_lon)
    mse_overall = euclidean_distance(true_lat, true_lon, overall_lat, overall_lon)
    mse_final_avg = euclidean_distance(true_lat, true_lon, final_avg_lat, final_avg_lon)

    # Add markers for each point with MSE values
    folium.Marker(
        location=[true_lat, true_lon],
        popup=f"True Location<br>MSE to SEGS: {mse_segs:.6f}<br>MSE to Overall: {mse_overall:.6f}<br>MSE to Final Avg: {mse_final_avg:.6f}",
        icon=folium.Icon(color="green", icon="ok-sign")
    ).add_to(m)

    folium.Marker(
        location=[segs_lat, segs_lon],
        popup=f"Averaged SEGS Prediction<br>MSE: {mse_segs:.6f}",
        icon=folium.Icon(color="orange", icon="cloud")
    ).add_to(m)

    folium.Marker(
        location=[overall_lat, overall_lon],
        popup=f"Overall Prediction<br>MSE: {mse_overall:.6f}",
        icon=folium.Icon(color="red", icon="cloud")
    ).add_to(m)

    folium.Marker(
        location=[final_avg_lat, final_avg_lon],
        popup=f"Final Averaged Prediction<br>MSE: {mse_final_avg:.6f}",
        icon=folium.Icon(color="blue", icon="star")
    ).add_to(m)

    # Draw lines connecting the points
    folium.PolyLine(
        locations=[[true_lat, true_lon], [segs_lat, segs_lon]],
        color="orange", weight=2, opacity=0.7, popup="True → SEGS Avg"
    ).add_to(m)

    folium.PolyLine(
        locations=[[true_lat, true_lon], [overall_lat, overall_lon]],
        color="red", weight=2, opacity=0.7, popup="True → Overall"
    ).add_to(m)

    folium.PolyLine(
        locations=[[segs_lat, segs_lon], [overall_lat, overall_lon]],
        color="purple", weight=2, opacity=0.7, popup="SEGS Avg → Overall"
    ).add_to(m)

    folium.PolyLine(
        locations=[[segs_lat, segs_lon], [final_avg_lat, final_avg_lon]],
        color="blue", weight=3, opacity=0.9, popup="SEGS Avg → Final Avg"
    ).add_to(m)

    folium.PolyLine(
        locations=[[overall_lat, overall_lon], [final_avg_lat, final_avg_lon]],
        color="blue", weight=3, opacity=0.9, popup="Overall → Final Avg"
    ).add_to(m)

# Save map to an HTML file
map_output = "predictions_map.html"
m.save(map_output)

print(f"Map saved to {map_output}")