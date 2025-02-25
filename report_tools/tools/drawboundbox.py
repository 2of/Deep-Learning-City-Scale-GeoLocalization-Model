import plotly.graph_objects as go
import json

# Load the bounding box configuration from the JSON file
with open('./config_box.json', 'r') as config_file:
    config = json.load(config_file)

# Extract the coordinates from the loaded config
N = config["N"]
S = config["S"]
E = config["E"]
W = config["W"]

# Define the corners of the bounding box using N, S, E, W
top_left = (N, W)
bottom_left = (S, W)
bottom_right = (S, E)
top_right = (N, E)

# Create the list of coordinates for the bounding box
bbox_coords = [top_left, (top_left[0], bottom_left[1]), bottom_left, bottom_right, 
               (top_right[0], bottom_right[1]), top_right, top_left]

# Extract latitude and longitude for the bounding box
latitudes = [coord[0] for coord in bbox_coords]
longitudes = [coord[1] for coord in bbox_coords]

# Create the map with Plotly
fig = go.Figure(go.Scattermapbox(
    mode='lines',
    lon=longitudes,
    lat=latitudes,
    line=dict(color='blue', width=2)  # Removed the fill option
))

# Update map layout
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=13,
    mapbox_center={"lat": (N + S) / 2, "lon": (W + E) / 2},
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

# Display the map
fig.show()