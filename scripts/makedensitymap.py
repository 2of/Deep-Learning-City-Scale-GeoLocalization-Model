import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the pickle file
pickle_file = './chicago.pkl'
df = pd.read_pickle(pickle_file)

# Extract latitude and longitude values
df['latitude'] = df['latitude']
df['longitude'] = df['longitude']

# Create a base map
map_center = [df['latitude'].mean(), df['longitude'].mean()]
base_map = folium.Map(location=map_center, zoom_start=13)

# Create a list of [latitude, longitude] for the heatmap
heat_data = [[row['latitude'], row['longitude']] for index, row in df.iterrows()]

# Add the heatmap to the base map with adjusted radius and blur
HeatMap(heat_data, radius=10, blur=6, max_zoom=13).add_to(base_map)

# Save the map to an HTML file
output_html = './heatmap_chicago.html'
base_map.save(output_html)

print(f"Heatmap saved to {output_html}")