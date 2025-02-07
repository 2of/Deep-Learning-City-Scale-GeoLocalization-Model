import os
import random
import tensorflow as tf





















FILE = "./embedded_datasets/UNSEEN/predictions21.tfrecord"

def parse(example_proto):
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

raw_data = tf.data.TFRecordDataset(FILE)
parsed_dataset = raw_data.map(parse)


for record in parsed_dataset:
    example_id = record['id'].numpy()
    print(f"Example ID: {example_id}")


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
        print("HELLO?")
        if not self.current_datasets:
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

        if len(latitudes) < self.batch_size:
            if self.current_file_index < len(self.tfrecord_files):
                self._load_next_files()
                return self.get_next_batch()
            else:
                # If there are no more files to load, return the current batch (even if it's smaller than batch_size)
                return (tf.convert_to_tensor(latitudes, dtype=tf.float32),
                        tf.convert_to_tensor(longitudes, dtype=tf.float32),
                        tf.convert_to_tensor(text_embeddings, dtype=tf.float32),
                        tf.convert_to_tensor(color_histograms, dtype=tf.float32),
                        tf.convert_to_tensor(class_names_vectors, dtype=tf.int64),
                        tf.convert_to_tensor(bboxes, dtype=tf.float32),
                        tf.convert_to_tensor(confidences, dtype=tf.float32))

        max_class_names_length = max(len(c) for c in class_names_vectors)
        max_bboxes_length = max(len(b) for b in bboxes)
        max_confidences_length = max(len(c) for c in confidences)

        class_names_vectors = [tf.pad(tf.convert_to_tensor(c, dtype=tf.int64), [[0, max_class_names_length - len(c)]], constant_values=0) for c in class_names_vectors]
        bboxes = [tf.pad(tf.convert_to_tensor(b, dtype=tf.float32), [[0, max_bboxes_length - len(b)]], constant_values=0.0) for b in bboxes]
        confidences = [tf.pad(tf.convert_to_tensor(c, dtype=tf.float32), [[0, max_confidences_length - len(c)]], constant_values=0.0) for c in confidences]

        return (tf.convert_to_tensor(latitudes, dtype=tf.float32),
                tf.convert_to_tensor(longitudes, dtype=tf.float32),
                tf.convert_to_tensor(text_embeddings, dtype=tf.float32),
                tf.convert_to_tensor(color_histograms, dtype=tf.float32),
                tf.convert_to_tensor(class_names_vectors, dtype=tf.int64),
                tf.convert_to_tensor(bboxes, dtype=tf.float32),
                tf.convert_to_tensor(confidences, dtype=tf.float32))

