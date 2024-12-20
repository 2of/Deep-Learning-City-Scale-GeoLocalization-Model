import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os

def list_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def view_shapefile_with_csv(shapefile_path, csv_path=None):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Plot the shapefile
    ax = gdf.plot(color='blue', edgecolor='black')
    
    # If a CSV path is provided, load and plot the points
    if csv_path:
        df = pd.read_csv(csv_path)
        plt.scatter(df['Longitude'], df['Latitude'], color='red', s=10, label='Points')
    
    plt.title(f"Shapefile: {shapefile_path}")
    if csv_path:
        plt.title(f"Shapefile: {shapefile_path} with Points from {csv_path}")
    plt.legend()
    plt.show()