import geopandas as gpd
import shapely.geometry as shp
import plotly.express as px
import pandas as pd

class Map:
    def __init__(self):
        print('Map Initialized')
    
    def render(self, center_lat, center_lon, use_csv=False, csv_path=None, radius=2000):  # Radius in meters
        # Create a GeoDataFrame with a point representing the city center
        gdf = gpd.GeoDataFrame(geometry=[shp.Point(center_lon, center_lat)], crs="EPSG:4326")
        
        # Project the point to UTM for accurate buffering in meters
        gdf = gdf.to_crs(epsg=32616)  # UTM Zone (appropriate for areas like Chicago)

        # Create a circular buffer around the point (buffer in meters)
        gdf['geometry'] = gdf.buffer(radius)  # Buffer distance in meters
        gdf = gdf.to_crs(epsg=4326)  # Reproject back to EPSG:4326 for Plotly visualization

        # Extract the coordinates of the buffer (circle) for plotting
        circle_coords = gdf.geometry[0].exterior.coords.xy
        circle_df = pd.DataFrame({'lon': circle_coords[0], 'lat': circle_coords[1]})

        # Create the map with Plotly
        fig = px.scatter_mapbox(circle_df, lat="lat", lon="lon", mapbox_style="carto-positron", zoom=12,
                                center={"lat": center_lat, "lon": center_lon})

        # Optionally, plot coordinates from a CSV file
        if use_csv and csv_path:
            coordinates_df = pd.read_csv(csv_path)
            fig.add_scattermapbox(lat=coordinates_df['Latitude'], lon=coordinates_df['Longitude'], mode='markers',
                                  marker=dict(size=6, color='blue'), name="Coordinates from CSV")

        # Plot the center point (city center)
        fig.add_scattermapbox(lat=[center_lat], lon=[center_lon], mode='markers', marker=dict(size=10, color='red'),
                              name="Center Point")

        # Plot the circular radius around the center
        fig.add_scattermapbox(lat=circle_df['lat'], lon=circle_df['lon'], mode='lines',
                              line=dict(width=2, color='black'), name="Radius")

        # Update layout settings to remove margins
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        # Show the plot
        fig.show()

# Example usage
a = Map()
# a.render(41.8781, -87.6298, radius=8000)  # Chicago city center with a 2 km radius
a.render(41.8781, -87.6298, True, './filtered_chicago_points.csv', radius=8000)
