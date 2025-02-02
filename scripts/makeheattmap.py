import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the CSV file
csv_file = './lat_lon_counts.csv'
df = pd.read_csv(csv_file)

# Parse the lat_lon_pair column to extract latitude and longitude
df['lat_lon_pair'] = df['lat_lon_pair'].apply(eval)
df['latitude'] = df['lat_lon_pair'].apply(lambda x: x[0])
df['longitude'] = df['lat_lon_pair'].apply(lambda x: x[1])

# Create a base map
map_center = [df['latitude'].mean(), df['longitude'].mean()]
base_map = folium.Map(location=map_center, zoom_start=13)

# Create a list of [latitude, longitude, count] for the heatmap
heat_data = [[row['latitude'], row['longitude'], row['count']] for index, row in df.iterrows()]

# Add the heatmap to the base map with adjusted radius and blur
HeatMap(heat_data, radius=3, blur=1).add_to(base_map)

# Save the map to an HTML file
output_html = './heatmap.html'
base_map.save(output_html)

print(f"Heatmap saved to {output_html}")