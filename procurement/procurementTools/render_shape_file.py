import geopandas as gpd
import matplotlib.pyplot as plt

# Load the shapefile into a GeoDataFrame
gdf = gpd.read_file("chicago_.shp")

# Plot the road network
gdf.plot(figsize=(10, 10), color='blue', linewidth=0.5)

# Show the plot
plt.title("Road Network in Chicago")
plt.show()
