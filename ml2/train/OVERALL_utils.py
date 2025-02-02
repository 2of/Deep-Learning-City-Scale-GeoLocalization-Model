import os
import random
import tensorflow as tf

class DataLoader:
    def __init__(self, data_dir, batch_size, shuffle=False, num_files_to_open=10, load_two_files=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_files_to_open = num_files_to_open
        self.load_two_files = load_two_files
        self.tfrecord_files = self._index_files()
        self.current_file_index = 0
        self.current_datasets = []

    def _index_files(self):
        tfrecord_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')]
        
        if self.shuffle:
            random.shuffle(tfrecord_files)
        
        return tfrecord_files

    def _parse_tf_example(self, example_proto):
        feature_description = {
            'id': tf.io.FixedLenFeature([], tf.int64),
            'latitude': tf.io.FixedLenFeature([], tf.float32),
            'longitude': tf.io.FixedLenFeature([], tf.float32),
            'text': tf.io.VarLenFeature(tf.string),
            'text_embeddings': tf.io.VarLenFeature(tf.float32),
            'color_histograms': tf.io.VarLenFeature(tf.float32),
            'num_detections': tf.io.FixedLenFeature([], tf.int64),
            'stacked_class_names_vector': tf.io.VarLenFeature(tf.int64),
            'stacked_bboxes': tf.io.VarLenFeature(tf.float32),
            'stacked_confidences': tf.io.VarLenFeature(tf.float32)
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    def _load_next_files(self):
        self.current_datasets = []
        files_to_open = 2 if self.load_two_files else 1
        for _ in range(files_to_open):
            if self.current_file_index >= len(self.tfrecord_files):
                break
            current_file = self.tfrecord_files[self.current_file_index]
            self.current_file_index += 1
            raw_dataset = tf.data.TFRecordDataset(current_file)
            parsed_dataset = raw_dataset.map(self._parse_tf_example)
            self.current_datasets.append(parsed_dataset)

    def get_next_batch(self):
        # if not self.current_datasets:
        self._load_next_files()
        
        latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = [], [], [], [], [], [], []
        
        for dataset in self.current_datasets:
            for parsed_record in dataset:
                latitude = parsed_record['latitude'].numpy()
                longitude = parsed_record['longitude'].numpy()
                text_embeddings.append(tf.sparse.to_dense(parsed_record['text_embeddings']).numpy().reshape((12, 128)))
                color_histograms.append(tf.sparse.to_dense(parsed_record['color_histograms']).numpy().reshape((3, 256)))
                
                # Handle variable length and empty tensors
                stacked_class_names_vector = tf.sparse.to_dense(parsed_record['stacked_class_names_vector']).numpy() if parsed_record['stacked_class_names_vector'].values.shape[0] > 0 else []
                stacked_bboxes = tf.sparse.to_dense(parsed_record['stacked_bboxes']).numpy() if parsed_record['stacked_bboxes'].values.shape[0] > 0 else []
                stacked_confidences = tf.sparse.to_dense(parsed_record['stacked_confidences']).numpy() if parsed_record['stacked_confidences'].values.shape[0] > 0 else []

                latitudes.append(latitude)
                longitudes.append(longitude)
                class_names_vectors.append(stacked_class_names_vector)
                bboxes.append(stacked_bboxes)
                confidences.append(stacked_confidences)

                if len(latitudes) >= self.batch_size:
                    break
            if len(latitudes) >= self.batch_size:
                break

        # Define the schema for padding
        max_detections = 24
        target_shapes = {
            "latitudes": (self.batch_size,),
            "longitudes": (self.batch_size,),
            "text_embeddings": (self.batch_size, 12, 128),
            "color_histograms": (self.batch_size, 3, 256),
            "class_names_vectors": (self.batch_size, max_detections),
            "bboxes": (self.batch_size, max_detections, 4),
            "confidences": (self.batch_size, max_detections)
        }

        # Pad the tensors to match the target shapes
        while len(latitudes) < target_shapes["latitudes"][0]:
            latitudes.append(0.0)
        
        while len(longitudes) < target_shapes["longitudes"][0]:
            longitudes.append(0.0)
        
        while len(text_embeddings) < target_shapes["text_embeddings"][0]:
            text_embeddings.append(tf.zeros((12, 128), dtype=tf.float32))
        
        while len(color_histograms) < target_shapes["color_histograms"][0]:
            color_histograms.append(tf.zeros((3, 256), dtype=tf.float32))
        
        while len(class_names_vectors) < target_shapes["class_names_vectors"][0]:
            class_names_vectors.append(tf.zeros(max_detections, dtype=tf.int64))
        
        while len(bboxes) < target_shapes["bboxes"][0]:
            bboxes.append(tf.zeros((max_detections, 4), dtype=tf.float32))
        
        while len(confidences) < target_shapes["confidences"][0]:
            confidences.append(tf.zeros(max_detections, dtype=tf.float32))

        # Ensure all lists are converted to tensors
        latitudes = tf.convert_to_tensor(latitudes, dtype=tf.float32)
        longitudes = tf.convert_to_tensor(longitudes, dtype=tf.float32)
        text_embeddings = tf.convert_to_tensor(text_embeddings, dtype=tf.float32)
        color_histograms = tf.convert_to_tensor(color_histograms, dtype=tf.float32)
        class_names_vectors = tf.convert_to_tensor(class_names_vectors, dtype=tf.int64)
        bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)
        confidences = tf.convert_to_tensor(confidences, dtype=tf.float32)

        return latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences
# Example usage
# data_dir = './data/tfrecords/lilMainBranc/'
# batch_size = 512
# data_loader = DataLoader(data_dir=data_dir,
#                          batch_size=batch_size,
#                          shuffle=True,
#                          load_two_files=True)

# try:
#     while True:
#         latitudes_batch, longitudes_batch, text_embeddings_batch, color_histograms_batch, class_names_vectors_batch, bboxes_batch, confidences_batch = data_loader.get_next_batch()

#         print(f"Latitudes Batch Shape: {latitudes_batch.shape}")
#         print(f"Longitudes Batch Shape: {longitudes_batch.shape}")
#         print(f"Text Embeddings Batch Shape: {text_embeddings_batch.shape}")
#         print(f"Color Histograms Batch Shape: {color_histograms_batch.shape}")
#         print(f"Class Names Vectors Batch Shape: {class_names_vectors_batch.shape}")
#         print(f"BBoxes Batch Shape: {bboxes_batch.shape}")
#         print(f"Confidences Batch Shape: {confidences_batch.shape}")
# except StopIteration:
#     print("All files have been processed.")
# # Example usage
# data_dir = './data/tfrecords/lilMainBranc/'
# batch_size = 512
# data_loader = DataLoader(data_dir=data_dir,
#                          batch_size=batch_size,
#                          shuffle=True,
#                          load_two_files=True)

# try:
#     while True:
#         latitudes_batch, longitudes_batch, text_embeddings_batch, color_histograms_batch, class_names_vectors_batch, bboxes_batch, confidences_batch = data_loader.get_next_batch()

#         print(f"Latitudes Batch Shape: {latitudes_batch.shape}")
#         print(f"Longitudes Batch Shape: {longitudes_batch.shape}")
#         print(f"Text Embeddings Batch Shape: {text_embeddings_batch.shape}")
#         print(f"Color Histograms Batch Shape: {color_histograms_batch.shape}")
#         print(f"Class Names Vectors Batch Shape: {class_names_vectors_batch.shape}")
#         print(f"BBoxes Batch Shape: {bboxes_batch.shape}")
#         print(f"Confidences Batch Shape: {confidences_batch.shape}")
# except StopIteration:
#     print("All files have been processed.")