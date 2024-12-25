import mercantile
import requests
import json
import os
import pandas as pd
from vt2geojson.tools import vt_bytes_to_geojson

def extract_feature_data(itemId, feature):
    """
    Extracts latitude, longitude, id, sequence_id, and is_pano from a GeoJSON feature.

    Args:
        feature (dict): A GeoJSON feature.

    Returns:
        dict: A dictionary containing feature data.
    """
    return {
        "itemId": itemId,
        "latitude": feature['geometry']['coordinates'][1],
        "longitude": feature['geometry']['coordinates'][0],
        "id": feature['properties']['id'],
        "sequence_id": feature['properties']['sequence_id'],
        "is_pano": feature['properties']['is_pano']
    }

def fetch_mapillary_data(west, south, east, north, output_file):
    print("GOT", west, south, east, north)
    
    with open('./KEYS.json', 'r') as key_file:
        keys = json.load(key_file)
        access_token = keys['mapillary']

    output = []
    tile_coverage = 'mly1_public'
    tile_layer = "image"
    tiles = list(mercantile.tiles(west, south, east, north, 14))
    print(f"Number of tiles: {len(tiles)}")
    
    for tile in tiles:
        print(f"Fetching tile: {tile}")
        tile_url = f'https://tiles.mapillary.com/maps/vtp/{tile_coverage}/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}'
        response = requests.get(tile_url)
        print(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Failed to fetch data for tile {tile.z}/{tile.x}/{tile.y}: {response.status_code}")
            continue

        data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=tile_layer)
        for feature in data['features']:
            output.append(extract_feature_data(tile.z * 1000000 + tile.x * 1000 + tile.y, feature))

    # Filter out points that are not in the bounding box
    filtered_output = [
        row for row in output 
        if west <= row["longitude"] <= east and south <= row["latitude"] <= north
    ]

    # Create a Pandas DataFrame
    df = pd.DataFrame(filtered_output)

    # Add additional columns with default values
    df['is_retrieved'] = False
    df['is_ingested'] = False

    # Save the DataFrame to a .pkl file
    os.makedirs('./data/GeoJSON', exist_ok=True)
    output_path = f'./data/GeoJSON/{output_file}'
    df.to_pickle(output_path)

    print(f"Data saved to: {output_path}")
    return df

if __name__ == "__main__":
    # Load bounding box coordinates from config_box.json
    with open('./config_box.json', 'r') as config_file:
        config = json.load(config_file)
    
    west = config['W']
    south = config['S']
    east = config['E']
    north = config['N']

    # Default output file name
    output_file = "image_data_chicago.pkl"

    print(f"Fetching Mapillary data for bounding box: {west}, {south}, {east}, {north}")
    print(f"Output will be saved to: ./data/GeoJSON/{output_file}")

    # Fetch data and return a DataFrame
    df = fetch_mapillary_data(west, south, east, north, output_file)

    # Display the first few rows of the DataFrame
    print(df.head())