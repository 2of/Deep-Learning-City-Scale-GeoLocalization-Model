import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from OVERALL_utils import DataLoader
import numpy as np
import csv

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize dictionaries to store MSE values
test_mses = {}
val_mses = {}
train_mses = {}

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            print(f"Epoch {epoch + 1}: loss = {logs.get('loss')}, mse = {logs.get('mse')}")
        else:
            print(f"Epoch {epoch + 1}: logs are None")

def randomize_endpoints(latitudes, longitudes, candidate_area_lat, candidate_area_lon, epoch, total_epochs):
    # we actually dont use this, think not great enoguh 
    factor = epoch / total_epochs
    latitudes += np.random.uniform(-candidate_area_lat * factor, candidate_area_lat * factor, size=latitudes.shape)
    longitudes += np.random.uniform(-candidate_area_lon * factor, candidate_area_lon * factor, size=longitudes.shape)
    
    # Clip to ensure values stay within the bounding box 
    latitudes = np.clip(latitudes, bounding_box['min_lat'], bounding_box['max_lat'])
    longitudes = np.clip(longitudes, bounding_box['min_lon'], bounding_box['max_lon'])
    
    return latitudes, longitudes

def normalize_coordinates(latitudes, longitudes, bounding_box):
    latitudes = (latitudes - bounding_box['min_lat']) / (bounding_box['max_lat'] - bounding_box['min_lat'])
    longitudes = (longitudes - bounding_box['min_lon']) / (bounding_box['max_lon'] - bounding_box['min_lon'])
    return latitudes, longitudes


def one_hot_encode_class_names(class_names_vectors, num_classes=12):
    # One-hot encode class names vectors using TensorFlow
    one_hot_encoded = tf.one_hot(class_names_vectors, depth=num_classes)
    return one_hot_encoded

def process_inputs_into_tensors(text_embeddings, color_histograms, class_names_vectors):
    
    class_names_one_hot = one_hot_encode_class_names(class_names_vectors)
    padding = [[0, 0], [0, 0], [0, 116]]  # Pad the last dimension from 12 to 128
    class_names_one_hot = tf.pad(class_names_one_hot, paddings=padding)
    color_histograms_reshaped = tf.reshape(color_histograms, (512, 3, 128, 2))
    color_histograms_resized = tf.reduce_mean(color_histograms_reshaped, axis=-1)
    
    
   # print(class_names_one_hot.shape , text_embeddings.shape, color_histograms_resized.shape)
    combined_tensor = tf.concat([class_names_one_hot, text_embeddings, color_histograms_resized], axis=1)
    #print("ME",combined_tensor.shape)
   # print(class_names_one_hot.shape)
    flattened_data = tf.reshape(combined_tensor, (combined_tensor.shape[0], -1))

    return(flattened_data)


def evaluate_clustering(latitudes, longitudes, cluster_centers, cluster_labels):
    # Calculate the mean latitude and longitude for each cluster
    true_coords = np.stack((latitudes, longitudes), axis=1)
    predicted_coords = np.zeros_like(true_coords)
    
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)
        cluster_mean = true_coords[cluster_indices].mean(axis=0)
        predicted_coords[cluster_indices] = cluster_mean
    
    mse = np.mean(np.square(true_coords - predicted_coords))
    return mse

def train_model(train_dir, val_dir, test_dir, batch_size, epochs, candidate_area_lat, candidate_area_lon):
    # Load the datasets using DataLoader
    print("Loading datasets...")
    train_loader = DataLoader(train_dir, batch_size, shuffle=True)
    val_loader = DataLoader(val_dir, batch_size, shuffle=False)
    test_loader = DataLoader(test_dir, batch_size, shuffle=False)
    print("Datasets loaded")

    # Set up the tensor tensor board TensorBoard callback
    log_dir = "./logs"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Learning rate scheduler with piecewise dropoff 
    def scheduler(epoch, lr):
        if epoch < 25:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    lr_scheduler = LearningRateScheduler(scheduler)

    # Open CSV files for writing MSEs
    with open('train_mses2.csv', 'w', newline='') as train_file, \
         open('val_mses2.csv', 'w', newline='') as val_file, \
         open('test_mses2.csv', 'w', newline='') as test_file:
        
        train_writer = csv.writer(train_file)
        val_writer = csv.writer(val_file)
        test_writer = csv.writer(test_file)

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            epoch_train_mses = []
            for batch in range(len(train_loader.tfrecord_files)):
                print(f"Training batch {batch + 1}/{len(train_loader.tfrecord_files)}")
                latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = train_loader.get_next_batch()
                
                # Normalize coordinates
                # latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
                
                # Randomize the initial ending points within the candidate area after warm-up phase
                if epoch >= 51:
                    latitudes, longitudes = randomize_endpoints(latitudes, longitudes, candidate_area_lat, candidate_area_lon, epoch, epochs)
                
                # Process inputs into tensors for clustering
                normalized_features = process_inputs_into_tensors(text_embeddings, color_histograms, class_names_vectors)
                
                # Apply K-Means clustering
                kmeans = KMeans(n_clusters=20)  # Adjust n_clusters as needed (10-50)
                kmeans.fit(normalized_features)
                
                # Get cluster labels and centroids
                cluster_labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_
                
                print(f"Batch {batch + 1} - Cluster Labels: {cluster_labels}, Cluster Centers: {cluster_centers}")
                
                mse_train = evaluate_clustering(latitudes, longitudes, cluster_centers, cluster_labels)
                epoch_train_mses.append(mse_train)

            train_writer.writerow(epoch_train_mses)

            # Validation
            epoch_val_mses = []
            for batch in range(len(val_loader.tfrecord_files)):
                print(f"Validating batch {batch + 1}/{len(val_loader.tfrecord_files)}")
                latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = train_loader.get_next_batch()
                
                
                # Normalize coordinates
                # latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
                
                # Process inputs into tensors for clustering
                normalized_features = process_inputs_into_tensors(text_embeddings, color_histograms, class_names_vectors)
                
                # Apply K-Means clustering
                kmeans = KMeans(n_clusters=20)  # Adjust n_clusters as needed (10-50)
                kmeans.fit(normalized_features)
                
                # Get cluster labels and centroids
                cluster_labels = kmeans.labels_
                cluster_centers = kmeans.cluster_centers_
                
                print(f"Batch {batch + 1} - Cluster Labels: {cluster_labels}, Cluster Centers: {cluster_centers}")
                
                mse_val = evaluate_clustering(latitudes, longitudes, cluster_centers, cluster_labels)
                epoch_val_mses.append(mse_val)

            val_writer.writerow(epoch_val_mses)

            # Custom callback
            CustomCallback().on_epoch_end(epoch)

        # Testing
        epoch_test_mses = []
        for batch in range(len(test_loader.tfrecord_files)):
            print(f"Testing batch {batch + 1}/{len(test_loader.tfrecord_files)}")
            latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = train_loader.get_next_batch()
            
            # Normalize coordinates
            # latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
            
            # Process inputs into tensors for clustering
            normalized_features = process_inputs_into_tensors(text_embeddings, color_histograms, class_names_vectors)
            
            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=20)  # Adjust n_clusters as needed (10-50)
            kmeans.fit(normalized_features)
            
            # Get cluster labels and centroids
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            
            print(f"Batch {batch + 1} - Cluster Labels: {cluster_labels}, Cluster Centers: {cluster_centers}")
            
            mse_test = evaluate_clustering(latitudes, longitudes, cluster_centers, cluster_labels)
            epoch_test_mses.append(mse_test)

if __name__ == "__main__":
    train_dir = "./embedded_datasets/MAIN_BRANCH_SHUFFLED/train"
    val_dir = "./embedded_datasets/MAIN_BRANCH_SHUFFLED/val"
    test_dir = "./embedded_datasets/MAIN_BRANCH_SHUFFLED/test" # Adjust these ones for sanity check and development
    batch_size = 512  # Adjust as needed
    epochs = 50  # Adjust as needed
    
    # Define the bounding box
    bounding_box = {'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332, 'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706}
    
    # Calculate the candidate area
    candidate_area_lat = (bounding_box['max_lat'] - bounding_box['min_lat']) / 2
    candidate_area_lon = (bounding_box['max_lon'] - bounding_box['min_lon']) / 2
    
    train_model(train_dir, val_dir, test_dir, batch_size, epochs, candidate_area_lat, candidate_area_lon)
