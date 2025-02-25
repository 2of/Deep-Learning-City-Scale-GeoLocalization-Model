# chat gpt



'''
Get bounding box from a .pkl file with lat long



handy if you're not logged in to gitihub and / or cant remember your dirs






Bounding Box: {'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332, 'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706}


'''

import pickle

def get_bounding_box(file_path, lat_col='latitude', lon_col='longitude'):
    """
    Opens a .pkl file and computes the bounding box based on latitude and longitude columns.

    :param file_path: Path to the .pkl file
    :param lat_col: Name of the column containing latitude values (default: 'latitude')
    :param lon_col: Name of the column containing longitude values (default: 'longitude')
    :return: A dictionary with the bounding box containing 'min_lat', 'max_lat', 'min_lon', 'max_lon'
    """
    # Load the .pkl file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Ensure the file contains the required columns
    if lat_col not in data or lon_col not in data:
        raise ValueError(f"The .pkl file does not contain '{lat_col}' and '{lon_col}' columns.")

    # Extract latitude and longitude
    latitudes = data[lat_col]
    longitudes = data[lon_col]

    # Compute the bounding box
    bounding_box = {
        'min_lat': min(latitudes),
        'max_lat': max(latitudes),
        'min_lon': min(longitudes),
        'max_lon': max(longitudes),
    }

    return bounding_box

# Example usage
if __name__ == "__main__":
    file_path = 'chicago.pkl'  # Replace with the actual file path
    try:
        bbox = get_bounding_box(file_path)
        print("Bounding Box:", bbox)
    except Exception as e:
        print("Error:", e)

