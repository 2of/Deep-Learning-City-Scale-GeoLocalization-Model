import os
import random
import tensorflow as tf
import sys 
import numpy as np


bounding_box = {'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332, 'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706}

# Calculate the candidate area
candidate_area_lat = (bounding_box['max_lat'] - bounding_box['min_lat']) / 2
candidate_area_lon = (bounding_box['max_lon'] - bounding_box['min_lon']) / 2


def randomize_endpoints(latitudes, longitudes, candidate_area_lat, candidate_area_lon, epoch, total_epochs):
    # Gradually increase randomization over epochs
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

def denormalize_coordinates(latitudes, longitudes, bounding_box):
    latitudes = latitudes * (bounding_box['max_lat'] - bounding_box['min_lat']) + bounding_box['min_lat']
    longitudes = longitudes * (bounding_box['max_lon'] - bounding_box['min_lon']) + bounding_box['min_lon']
    return latitudes, longitudes

def adjust_coordinates(latitudes, longitudes):
    # latitudes -= 41.9
    # longitudes += 87.6
    return latitudes, longitudes

def latitude_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true[:, 0], y_pred[:, 0])

def longitude_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true[:, 1], y_pred[:, 1])

def combined_loss(y_true, y_pred):
    lat_loss = latitude_loss(y_true, y_pred)
    lon_loss = longitude_loss(y_true, y_pred)
    return lat_loss + lon_loss  # You can adjust the weights if needed

def calculate_geodesic_distance(y_true, y_pred):
    distances = []
    for i in range(len(y_true)):
        true_coords = (y_true[i][0], y_true[i][1])
        pred_coords = (y_pred[i][0], y_pred[i][1])
        try:
            distances.append(geodesic(true_coords, pred_coords).meters)
        except:
            distances.append(0)
    return np.mean(distances), np.median(distances)

def calculate_directional_error(y_true, y_pred):
    try:
        errors = []
        for i in range(len(y_true)):
            true_vector = [y_true[i][0], y_true[i][1]]
            pred_vector = [y_pred[i][0], y_pred[i][1]]
            errors.append(cosine(true_vector, pred_vector))
        return np.mean(errors)
    except:
        return 0
    
    
    
class DataLoader:
    def __init__(self, data_dir, batch_size, shuffle=False, num_files_to_open=10, load_two_files=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_files_to_open = num_files_to_open
        self.load_two_files = load_two_files
        self.tfrecord_files = self._index_files()
        print(f"Total number of TFRecord files: {len(self.tfrecord_files)}")  # Print total files
        self.current_file_index = 0
        self.current_datasets = []

    def _index_files(self):
        tfrecord_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')]
        
        if self.shuffle:
            random.shuffle(tfrecord_files)
        
        return tfrecord_files

    def _parse_tf_example(self,example_proto):
        feature_description = {
            'id': tf.io.FixedLenFeature([], tf.int64),
            'latitude': tf.io.FixedLenFeature([], tf.float32),
            'longitude': tf.io.FixedLenFeature([], tf.float32),
            'text_embeddings': tf.io.VarLenFeature(tf.float32),
            'color_histogram': tf.io.VarLenFeature(tf.float32)
        }
        return tf.io.parse_single_example(example_proto, feature_description)


    def _load_next_files(self):
        self.current_datasets = []
        files_to_open = 2 if self.load_two_files else 1
        print("LOAD LOAD LOAD ")
        
        current_file = self.tfrecord_files[self.current_file_index]
        print(f"Loading file {current_file} (index {self.current_file_index})")
        raw_dataset = tf.data.TFRecordDataset(current_file)
        parsed_dataset = raw_dataset.map(self._parse_tf_example)
        self.current_datasets.append(parsed_dataset)

    def get_next_batch(self):
        if self.current_file_index >= len(self.tfrecord_files):
            print(f"End of files reached: current_file_index={self.current_file_index}, len(self.tfrecord_files)={len(self.tfrecord_files)}")
            return None
        
        self._load_next_files()
        
        latitudes, longitudes, text_embeddings, color_histograms, num_detections = [], [], [], [], []
        
        for dataset in self.current_datasets:
            for parsed_record in dataset:
            #    print(parsed_record)
                latitude = parsed_record['latitude'].numpy()
                longitude = parsed_record['longitude'].numpy()
                text = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy().reshape((12, 128))
                color = tf.sparse.to_dense(parsed_record['color_histogram']).numpy().reshape(( 256))
                # num_detection = parsed_record['num_detections'].numpy()
                # print(num_detection)
                
                # # Reshape color_histogram and text_embedding based on num_detections
                # color_histogram = color_histogram.reshape((num_detection, 256))
                # text_embedding = text_embedding.reshape((num_detection, 12, 128))
                
                latitudes.append(latitude)
                longitudes.append(longitude)
                text_embeddings.append(text)
                color_histograms.append(color)
                # num_detections.append(num_detection)

        self.current_file_index += 1

        return (tf.convert_to_tensor(latitudes, dtype=tf.float32),
                tf.convert_to_tensor(longitudes, dtype=tf.float32),
                tf.ragged.constant(text_embeddings, dtype=tf.float32).to_tensor(),
                tf.ragged.constant(color_histograms, dtype=tf.float32).to_tensor())
                # tf.convert_to_tensor(num_detections, dtype=tf.int64))

# # Example usage
# data_dir = './embedded_datasets/SEGS_SINGLE_PER_ROW/test'
# batch_size = 512

# data_loader = DataLoader(data_dir=data_dir,
#                          batch_size=batch_size,
#                          shuffle=False,
#                          load_two_files=False)

# try:
#     while True:
#         batch = data_loader.get_next_batch()
#         if batch is None:
#             break
#         latitudes_batch, longitudes_batch, text_embeddings_batch, color_histograms_batch = batch
#         # print(f"Batch: Latitudes size: {len(latitudes_batch)}, Longitudes size: {len(longitudes_batch)}")
#         # print(latitudes_batch)
# except SystemExit:
#     print("All files have been processed.")
# Example usage
# data_dir = './embedded_datasets/SEGS_SINGLE_PER_ROW/truesmall'
# batch_size = 512

# data_loader = DataLoader(data_dir=data_dir,
#                          batch_size=batch_size,
#                          shuffle=False,
#                          load_two_files=False)

# try:
#     while True:
#         batch = data_loader.get_next_batch()
#         if batch is None:
#             break
#         latitudes_batch, longitudes_batch, text_embeddings_batch, color_histograms_batch = batch
#         print(f"Batch: Latitudes size: {len(latitudes_batch)}, Longitudes size: {len(longitudes_batch)}")
#         print(latitudes_batch)
# except SystemExit:
#     print("All files have been processed.")
