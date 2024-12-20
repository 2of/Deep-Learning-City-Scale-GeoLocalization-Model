import shapefile
import csv
from geopy.distance import geodesic
from rtree import index
import os

def extract_points_sample(shapefile_path, output_csv_base, distance=2):
    # Read the shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Create a spatial index
    idx = index.Index()
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv_base)
    os.makedirs(output_dir, exist_ok=True)
    
    file_count = 0
    line_count = 0
    
    # Open the first CSV file for writing
    csvfile = open(f'{output_csv_base}_{file_count}.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Latitude', 'Longitude'])
    
    # Process all shapes in the shapefile
    for shape in sf.shapes():
        points = shape.points
        
        # Write all points, including intermediate points, to the CSV file
        for i in range(len(points) - 1):
            # Write the current point
            lat, lon = round(points[i][1], 6), round(points[i][0], 6)
            csvwriter.writerow([lat, lon])
            line_count += 1
            
            if line_count >= 1000:
                csvfile.close()
                file_count += 1
                line_count = 0
                csvfile = open(f'{output_csv_base}_{file_count}.csv', 'w', newline='')
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Latitude', 'Longitude'])
            
            dist = geodesic(points[i], points[i + 1]).meters
            
            if dist > distance:
                # Calculate intermediate points
                num_intermediate_points = int(dist // distance)
                for j in range(1, num_intermediate_points + 1):
                    fraction = j * distance / dist
                    lat = round(points[i][1] + fraction * (points[i + 1][1] - points[i][1]), 6)
                    lon = round(points[i][0] + fraction * (points[i + 1][0] - points[i][0]), 6)
                    csvwriter.writerow([lat, lon])
                    line_count += 1
                    
                    if line_count >= 1000:
                        csvfile.close()
                        file_count += 1
                        line_count = 0
                        csvfile = open(f'{output_csv_base}_{file_count}.csv', 'w', newline='')
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(['Latitude', 'Longitude'])
        
        # Write the last point
        lat, lon = round(points[-1][1], 6), round(points[-1][0], 6)
        csvwriter.writerow([lat, lon])
        line_count += 1
        
        if line_count >= 1000:
            csvfile.close()
            file_count += 1
            line_count = 0
            csvfile = open(f'{output_csv_base}_{file_count}.csv', 'w', newline='')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Latitude', 'Longitude'])
    
    # Close the last CSV file

    csvfile.close()
def extract_points_from_shapefile(shapefile_path, output_csv, distance=2):
    # Read the shapefile
    sf = shapefile.Reader(shapefile_path)
    
    # Create a spatial index
    idx = index.Index()
    
    
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['Latitude', 'Longitude'])
        
        point_id = 0
        
        # Traverse each shape (road) in the shapefile
        total_len = len(sf.shapes())
        ith = 0
        for shape in sf.shapes():
            print("doing ", ith, "of " , total_len)
            ith += 1
            points = shape.points
            if not points:
                continue
            
            # Write the first point
            lat, lon = points[0]
            csvwriter.writerow([lat, lon])
            idx.insert(point_id, (lon, lat, lon, lat))
            last_point = points[0]
            point_id += 1
            
            # Write points at specified distance intervals
            # of = len(points)
            # ith = 0
            for point in points[1:]:
                # ith += 1
                # print("doing line" , ith , "of", of)
                while geodesic(last_point, point).km >= distance:
                    # Calculate intermediate point
                    fraction = distance / geodesic(last_point, point).km
                    lat = last_point[0] + fraction * (point[0] - last_point[0])
                    lon = last_point[1] + fraction * (point[1] - last_point[1])
                    intermediate_point = (lat, lon)
                    
                    if list(idx.intersection((lon, lat, lon, lat))) == []:
                        csvwriter.writerow([lat, lon])
                        idx.insert(point_id, (lon, lat, lon, lat))
                        last_point = intermediate_point
                        point_id += 1
                    else:
                        break
                
                last_point = point


def print_points_from_shapefile(shapefile_path, distance=2):
    # Load the shapefile

    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Ensure the geometry is LineString
    gdf = gdf[gdf.geometry.type == 'LineString']

    # Extract points along the lines
    points = []
    for line in gdf.geometry:
        if isinstance(line, LineString):
            num_points = int(line.length / distance) + 1
            for i in range(num_points):
                point = line.interpolate(i * distance)
                print(point)
                points.append((point.y, point.x))  # (latitude, longitude)

