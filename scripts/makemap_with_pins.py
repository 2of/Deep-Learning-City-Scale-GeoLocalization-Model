import pandas as pd
import folium

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

# Define the coordinates for the two pins
pin1_coords = (41.902433, -87.627463)  # Coordinates for pin 1
pin2_coords = (41.902433, -87.679728)  # Coordinates for pin 2

# Add the pins to the map with labels always visible
folium.Marker(location=pin1_coords, icon=folium.Icon(icon='info-sign')).add_to(base_map)
folium.Marker(location=pin2_coords, icon=folium.Icon(icon='info-sign')).add_to(base_map)

# Add text labels next to the pins with styled HTML for better readability
folium.Marker(
    location=pin1_coords,
    icon=folium.DivIcon(html=f'''
        <div style="
            font-size: 24px;
            color: black;
            background-color: white;
            padding: 5px;
            margin: 12px;
            width: 280px;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
            backdrop-filter: blur(5px);
        ">
          {pin1_coords}
        </div>
    ''')
).add_to(base_map)

folium.Marker(
    location=pin2_coords,
    icon=folium.DivIcon(html=f'''
        <div style="
            font-size: 24px;
            color: black;
            background-color: white;
            margin: 12px;
            padding: 5px;
            width: 280px;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
            backdrop-filter: blur(5px);
        ">
 {pin2_coords}
        </div>
    ''')
).add_to(base_map)

# Define the bounding box coordinates
bounding_box = {'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332, 'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706}

# Add the bounding box to the map
folium.Rectangle(
    bounds=[
        [bounding_box['min_lat'], bounding_box['min_lon']],
        [bounding_box['max_lat'], bounding_box['max_lon']]
    ],
    color='red',
    fill=True,
    fill_opacity=0.1
).add_to(base_map)

# Calculate the distance in terms of the decimal difference in longitudes
distance = abs(pin1_coords[1] - pin2_coords[1])

# Add a line between the two points with a label showing the distance
folium.PolyLine(
    locations=[pin1_coords, pin2_coords],
    color='blue',
    weight=2.5,
    opacity=1
).add_to(base_map)

# Add a label for the distance
mid_point = [(pin1_coords[0] + pin2_coords[0]) / 2, (pin1_coords[1] + pin2_coords[1]) / 2]
folium.Marker(
    location=mid_point,
    icon=folium.DivIcon(html=f'''
        <div style="
            font-size: 24px;
            color: black;
            background-color: white;
            padding: 5px;
            width: 280px;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
            backdrop-filter: blur(5px);
        ">
            Distance (Longitude): {round(distance, 6)}
        </div>
    ''')
).add_to(base_map)

# Save the map to an HTML file
output_html = './map_with_pins_and_bounding_box.html'
base_map.save(output_html)

print(f"Map with pins and bounding box saved to {output_html}")