import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import json



from haversine import haversine

def calculate_statistics(kmeans_model, data, latitudes, longitudes):
    predicted_clusters = kmeans_model.predict(data)
    cluster_averages = {}
    mse = 0
    msd = 0
    intra_cluster_distances = []
    avg_geo_distances = []  # Store average geographic distances

    for cluster in range(kmeans_model.n_clusters):
        cluster_indices = np.where(predicted_clusters == cluster)
        cluster_latitudes = latitudes[cluster_indices]
        cluster_longitudes = longitudes[cluster_indices]
        cluster_data = data[cluster_indices]

        # Geographic centroid of the cluster
        avg_latitude = np.mean(cluster_latitudes)
        avg_longitude = np.mean(cluster_longitudes)
        cluster_center = kmeans_model.cluster_centers_[cluster]

        # Mean Squared Error in the feature space
        mse += np.sum((cluster_data - cluster_center) ** 2)

        # Mean Squared Distance
        distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
        msd += np.sum(distances ** 2)

        # Intra-cluster distances
        intra_cluster_distance = np.mean(distances)
        intra_cluster_distances.append(intra_cluster_distance)

        # Calculate geographic distances
        geographic_distances = [
            haversine((lat, lon), (avg_latitude, avg_longitude))
            for lat, lon in zip(cluster_latitudes, cluster_longitudes)
        ]
        avg_geo_distance = np.mean(geographic_distances)
        avg_geo_distances.append(avg_geo_distance)

        cluster_averages[cluster] = {
            "avg_latitude": float(avg_latitude),
            "avg_longitude": float(avg_longitude),
            "num_points": len(cluster_indices[0]),
            "mean_distance": float(np.mean(distances)),  # Mean feature space distance
            "avg_geo_distance_km": float(avg_geo_distance),  # Mean geographic distance in km
        }

    mse /= len(data)  # Average MSE
    msd /= len(data)  # Average Mean Squared Distance
    avg_intra_distance = np.mean(intra_cluster_distances)

    return {
        "cluster_averages": cluster_averages,
        "mse": float(mse),
        "msd": float(msd),
        "avg_intra_distance": float(avg_intra_distance),
        "avg_geo_distances": float(np.mean(avg_geo_distances)),  # Overall avg geo distance
    }
    
    
    
def parse_tfrecord_fn(example):
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'color_histograms': tf.io.VarLenFeature(tf.float32),
        'num_detections': tf.io.FixedLenFeature([], tf.int64),
        'text_embeddings': tf.io.VarLenFeature(tf.float32),
        'stacked_class_names_vector': tf.io.VarLenFeature(tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)

def one_hot_encode(vector, num_classes):
    one_hot = np.zeros((len(vector), num_classes))
    for i, class_id in enumerate(vector):
        one_hot[i, class_id - 1] = 1
    return one_hot

def load_and_prepare_data(tfrecord_filename, num_classes, max_length=None):
    print("Loading and parsing TFRecord file...")
    raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    
    data = []
    latitudes = []
    longitudes = []
    
    print("Extracting and arranging features...")
    for i, parsed_record in enumerate(parsed_dataset):
        latitude = parsed_record['latitude'].numpy()
        longitude = parsed_record['longitude'].numpy()
        text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
        stacked_class_names_vector = tf.sparse.to_dense(parsed_record['stacked_class_names_vector']).numpy()
        stacked_class_names_one_hot = one_hot_encode(stacked_class_names_vector, num_classes)
        feature_vector = np.concatenate((text_embeddings, stacked_class_names_one_hot.flatten()))
        data.append(feature_vector)
        latitudes.append(latitude)
        longitudes.append(longitude)
    
    print("Feature extraction complete. Total records processed:", len(data))
    
    if max_length is None:
        max_length = max(len(vec) for vec in data)
    padded_data = np.array([np.pad(vec, (0, max_length - len(vec)), 'constant') for vec in data])
    
    # Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(padded_data)
    
    return normalized_data, np.array(latitudes), np.array(longitudes), max_length

def train_kmeans(data, n_clusters):
    print(f"Training k-means model for k = {n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    print("Training complete.")
    return kmeans


def save_results_to_json(k, results):
    filename = f"cluster_results_k_{k}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results for k = {k} saved to {filename}.")

def find_convergence(mse_values):
    if len(mse_values) < 2:
        return None
    deltas = [mse_values[i - 1] - mse_values[i] for i in range(1, len(mse_values))]
    return deltas

if __name__ == "__main__":
    tfrecord_filename = "./MAIN_BRANCH_SHUFFLED/all_train.tfrecord"
    num_classes = 10  # Set the number of classes for one-hot encoding
    k_range = range(5, 21)  # K from 5 to 20

    data, latitudes, longitudes, max_length = load_and_prepare_data(tfrecord_filename, num_classes)

    mse_values = []
    for k in k_range:
        kmeans_model = train_kmeans(data, n_clusters=k)
        results = calculate_statistics(kmeans_model, data, latitudes, longitudes)
        # other_results = calculate_statistics
        save_results_to_json(k, results)
        mse_values.append(results["mse"])

    convergence = find_convergence(mse_values)
    print("\nMSE Convergence Trend:")
    for k, mse in zip(k_range, mse_values):
        print(f"k = {k}, MSE = {mse}")