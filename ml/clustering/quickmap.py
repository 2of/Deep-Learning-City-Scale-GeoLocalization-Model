import folium
import json

def plot_clusters_from_json(json_filename, map_filename="clusters_map.html"):
    # Load cluster data from JSON file
    with open(json_filename, "r") as f:
        cluster_data = json.load(f)

    # Extract cluster averages (latitude and longitude)
    cluster_averages = cluster_data.get("cluster_averages", {})
    if not cluster_averages:
        print("No cluster averages found in the JSON file.")
        return

    # Compute the map center as the average latitude and longitude of all clusters
    latitudes = [info["avg_latitude"] for info in cluster_averages.values()]
    longitudes = [info["avg_longitude"] for info in cluster_averages.values()]
    map_center = [sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)]

    # Create a map centered around the average location
    mymap = folium.Map(location=map_center, zoom_start=15)

    # Add markers for each cluster
    for cluster_id, cluster_info in cluster_averages.items():
        folium.Marker(
            [cluster_info["avg_latitude"], cluster_info["avg_longitude"]],
            popup=f"Cluster {cluster_id}<br>Points: {cluster_info['num_points']}<br>Mean Distance: {cluster_info['mean_distance']:.2f}",
        ).add_to(mymap)

    # Save the map to an HTML file
    mymap.save(map_filename)
    print(f"Map with cluster locations has been saved to {map_filename}")

# Usage
json_filename = "./cluster_results_k_20.json"  # Replace with your actual JSON filename
plot_clusters_from_json(json_filename)