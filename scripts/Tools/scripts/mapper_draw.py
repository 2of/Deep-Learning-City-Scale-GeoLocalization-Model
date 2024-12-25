import geopandas as gpd
import shapely.geometry as shp
import plotly.express as px
import pandas as pd

# Chicago city center latitude and longitude
lat, lon = 42.29863521978824, -83.12654700254598,
radius = 2  # 8 km radius in meters

# Create a GeoDataFrame with a Point representing the city center
gdf = gpd.GeoDataFrame(geometry=[shp.Point(lon, lat)], crs="EPSG:4326")

# Project to a local coordinate system (UTM Zone 16N) to create an accurate buffer in meters
gdf = gdf.to_crs(epsg=32616)  # UTM zone 16N for Chicago
gdf['geometry'] = gdf.buffer(radius)  # Create an 8 km buffer around the city center

# Re-project back to EPSG:4326 for plotting in Plotly
gdf = gdf.to_crs(epsg=4326)

# Extract the coordinates of the circular buffer for plotting
circle_coords = gdf.geometry[0].exterior.coords.xy
circle_df = pd.DataFrame({'lon': circle_coords[0], 'lat': circle_coords[1]})

# Load points from 'coordinates.csv' file
coordinates_df = pd.read_csv('coordinates.csv')
print(coordinates_df)
# Create the map with Plotly
fig = px.scatter_mapbox(circle_df, lat="lat", lon="lon", mapbox_style="carto-positron", zoom=11,
                        center={"lat": lat, "lon": lon})

# Add the city center as a separate point
fig.add_scattermapbox(lat=[lat], lon=[lon], mode='markers', marker=dict(size=10, color='red'),
                      name="City Center")

# Add the circular radius as a line around the center
fig.add_scattermapbox(lat=circle_df['lat'], lon=circle_df['lon'], mode='lines',
                      line=dict(width=2, color='black'), name="Radius")

# Plot points from 'coordinates.csv' file
fig.add_scattermapbox(lat=coordinates_df['Latitude'], lon=coordinates_df['Longitude'], mode='markers',
                      marker=dict(size=6, color='blue'), name="Coordinates from CSV")

# Display the map
fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
fig.show()
