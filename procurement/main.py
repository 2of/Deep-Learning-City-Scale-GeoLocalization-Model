import json
import os
import sys
from tools.generate_csv import generate_points
from tools.download_road_network import download_map
from tools.extract_roads_from_shapefile_to_csv import extract_points_sample,extract_points_from_shapefile, print_points_from_shapefile
from tools.display import view_shapefile_with_csv

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def select_city(config):
    cities = list(config['cities'].keys())
    print("Select a city:")
    for i, city in enumerate(cities, 1):
        print(f"{i}. {city}")
    choice = int(input("Enter the number of the city: "))
    selected_city = cities[choice - 1]
    return config['cities'][selected_city]

def list_shapefiles(directory):
    shapefiles = [f for f in os.listdir(directory) if f.endswith('.shp')]
    if not shapefiles:
        print("No shapefiles found in the directory.")
        return None
    print("Select a shapefile:")
    for i, shapefile in enumerate(shapefiles, 1):
        print(f"{i}. {shapefile}")
    choice = int(input("Enter the number of the shapefile: "))
    return shapefiles[choice - 1]

def list_csv_files(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the directory.")
        return None
    print("Select a CSV file (optional):")
    print("0. None")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. {csv_file}")
    choice = int(input("Enter the number of the CSV file (or 0 for none): "))
    if choice == 0:
        return None
    return csv_files[choice - 1]

def main():
    config_file = 'config.json'
    config = load_config(config_file)
    
    while True:
        print("\nOptions:")
        print("1. Generate Points")
        print("2. Download Shapefile")
        print("3. Extract Points from Shapefile")
        print("4. Just Print Lat/Long from Roads on a Shapefile (Warning: Lots of Text!)")
        print("5. View Shapefile with Optional CSV Points")
        print("9. Exit")
        choice = int(input("Enter your choice: "))
        
        if choice == 1:
            city_config = select_city(config)
            center_lat = city_config['latitude']
            center_lon = city_config['longitude']
            radius_km = float(input("Enter the radius in km: "))
            distance = float(input("Enter the distance between points in meters: "))
            output_file_prefix = input("Enter the output file prefix: ")
            generate_points(center_lat, center_lon, radius_km, distance, output_file_prefix)
        
        elif choice == 2:
            city_config = select_city(config)
            place_name = city_config['shapefile_place_name']
            save_path = input("Enter the save path (default is 'data/shapefiles'): ") or 'data/shapefiles'
            download_map(place_name, save_path)
        
        elif choice == 3:
            shapefile_directory = 'data/shapefiles'
            shapefile_name = list_shapefiles(shapefile_directory)
            if shapefile_name:
                shapefile_path = os.path.join(shapefile_directory, shapefile_name)
                output_csv = os.path.join(shapefile_directory, shapefile_name.replace('.shp', '_roads_only.csv'))
                distance = float(input("Enter the distance between points in meters: "))
                extract_points_sample(shapefile_path, output_csv, distance)
        
        elif choice == 4:
            shapefile_directory = 'data/shapefiles'
            shapefile_name = list_shapefiles(shapefile_directory)
            if shapefile_name:
                shapefile_path = os.path.join(shapefile_directory, shapefile_name)
                distance = float(input("Enter the distance between points in meters: "))
                print_points_from_shapefile(shapefile_path, distance)
        
        elif choice == 5:
            shapefile_directory = 'data/shapefiles'
            shapefile_name = list_shapefiles(shapefile_directory)
            if shapefile_name:
                shapefile_path = os.path.join(shapefile_directory, shapefile_name)
                csv_name = list_csv_files(shapefile_directory)
                if csv_name:
                    csv_path = os.path.join(shapefile_directory, csv_name)
                    view_shapefile_with_csv(shapefile_path, csv_path)
                else:
                    view_shapefile_with_csv(shapefile_path)
        
        elif choice == 9:
            print("Exiting...")
            sys.exit()
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
