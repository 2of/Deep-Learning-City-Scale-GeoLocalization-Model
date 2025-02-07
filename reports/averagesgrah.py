import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the averaged predictions CSV
file_path = 'averaged_predictions.csv'
df = pd.read_csv(file_path)

# Take a random sample of 10 rows for better visualization
df = df.sample(10, random_state=42)

# Bounding box values
bounding_box = {
    'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332,
    'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706
}

# Function to denormalize coordinates
def denormalize_coordinates(lat, lon, bounding_box):
    lat = float(lat) * (bounding_box['max_lat'] - bounding_box['min_lat']) + bounding_box['min_lat']
    lon = float(lon) * (bounding_box['max_lon'] - bounding_box['min_lon']) + bounding_box['min_lon']
    return lat, lon

# Function to compute Euclidean distance (MSE proxy)
def euclidean_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

# Denormalize all points in the dataframe
for col in ["True Latitude", "Averaged Pred Latitude (SEGS)", "Pred Latitude (Overall)", "Final Averaged Latitude"]:
    df[col], df[col.replace("Latitude", "Longitude")] = zip(*df.apply(
        lambda row: denormalize_coordinates(row[col], row[col.replace("Latitude", "Longitude")], bounding_box),
        axis=1
    ))

# Calculate Euclidean distances
df["MSE_SEGS"] = df.apply(lambda row: euclidean_distance(row["True Latitude"], row["True Longitude"],
                                                          row["Averaged Pred Latitude (SEGS)"], row["Averaged Pred Longitude (SEGS)"]), axis=1)

df["MSE_Overall"] = df.apply(lambda row: euclidean_distance(row["True Latitude"], row["True Longitude"],
                                                             row["Pred Latitude (Overall)"], row["Pred Longitude (Overall)"]), axis=1)

df["MSE_Final_Avg"] = df.apply(lambda row: euclidean_distance(row["True Latitude"], row["True Longitude"],
                                                               row["Final Averaged Latitude"], row["Final Averaged Longitude"]), axis=1)

# Plot the MSE values
plt.figure(figsize=(10, 5))

# Plot each category of MSE
plt.plot(df.index, df["MSE_SEGS"], marker='o', linestyle='-', color='orange', label="MSE to SEGS")
plt.plot(df.index, df["MSE_Overall"], marker='s', linestyle='-', color='red', label="MSE to Overall")
plt.plot(df.index, df["MSE_Final_Avg"], marker='^', linestyle='-', color='blue', label="MSE to Final Avg")

# Labels and legend
plt.xlabel("Data Point Index")
plt.ylabel("Euclidean Distance (MSE Proxy)")
plt.title("MSE Distance from True Location to Predictions")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()