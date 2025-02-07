import os
import random
import tensorflow as tf

# Quick cut down versio nof data loader to write records to random files based on num_bins
# Reads all files in a tfrecords directory, randomly assigns a bin to each and just writes those out 
# to a new file when bin size met
# should be random enough to just choose records files for val and test with no issue
# THIS ONE only for OVERALL, not segs. See 'shuffler_segs' for that one. 


class DataLoader_OVERALL:
    def __init__(self, data_dir, output_dir, batch_size=512, num_bins=75):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_bins = num_bins
        self.tfrecord_files = self._index_files()
        self.buffers = [[] for _ in range(num_bins)]
        self.buffer_counts = [0] * num_bins
        self.file_counts = [0] * num_bins

    def _index_files(self):
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')]

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

    def _write_buffer_to_file(self, bin_index):
        output_file = os.path.join(self.output_dir, f'shuffled_{bin_index}_{self.file_counts[bin_index]}.tfrecord')
        with tf.io.TFRecordWriter(output_file) as writer:
            for record in self.buffers[bin_index]:
                writer.write(record)
        print(f"Written {len(self.buffers[bin_index])} records to {output_file}")
        self.buffers[bin_index] = []
        self.buffer_counts[bin_index] = 0
        self.file_counts[bin_index] += 1

    def shuffle_and_write(self):
        for tfrecord_file in self.tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            for raw_record in raw_dataset:
                bin_index = random.randint(0, self.num_bins - 1)
                self.buffers[bin_index].append(raw_record.numpy())
                self.buffer_counts[bin_index] += 1
                if self.buffer_counts[bin_index] >= self.batch_size:
                    self._write_buffer_to_file(bin_index)
            print(f"After reading {tfrecord_file}:")
            for i in range(self.num_bins):
                print(f"  Bin {i} has {self.buffer_counts[i]} records")
        
        # Write remaining records in buffers
        for bin_index in range(self.num_bins):
            if self.buffer_counts[bin_index] > 0:
                self._write_buffer_to_file(bin_index)

# Example usage
# data_dir = './data/tfrecords/MAIN_BRANCH'
# output_dir = './data/tfrecords/MAIN_BRANCH_SHUFFLED/'
# data_loader = DataLoader(data_dir=data_dir, output_dir=output_dir)
# data_loader.shuffle_and_write()


import os
import random
import tensorflow as tf

# Quick cut down versio nof data loader to write records to random files based on num_bins
# Reads all files in a tfrecords directory, randomly assigns a bin to each and just writes those out 
# to a new file when bin size met
# should be random enough to just choose records files for val and test with no issue
# THIS ONE only for OVERALL, not segs. See 'shuffler_segs' for that one. 


import os
import random
import tensorflow as tf

class DataLoader_segs:
    def __init__(self, data_dir, output_dir, batch_size=512, num_bins=50):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_bins = num_bins
        self.tfrecord_files = self._index_files()
        print(f"Indexed TFRecord files: {self.tfrecord_files}")
        self.buffers = [[] for _ in range(num_bins)]
        self.buffer_counts = [0] * num_bins
        self.file_counts = [0] * num_bins

    def _index_files(self):
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.tfrecord')]
        print(f"Found {len(files)} TFRecord files.")
        return files

    def _parse_tf_example(self, example_proto):
        feature_description = {
            'id': tf.io.FixedLenFeature([], tf.int64),
            'latitude': tf.io.FixedLenFeature([], tf.float32),
            'longitude': tf.io.FixedLenFeature([], tf.float32),
            'text': tf.io.VarLenFeature(tf.string),
            'text_embeddings': tf.io.VarLenFeature(tf.float32),
            'color_histograms': tf.io.VarLenFeature(tf.float32),
            'num_detections': tf.io.FixedLenFeature([], tf.int64)
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    def _write_buffer_to_file(self, bin_index):
        output_file = os.path.join(self.output_dir, f'shuffled_{bin_index}_{self.file_counts[bin_index]}.tfrecord')
        with tf.io.TFRecordWriter(output_file) as writer:
            for record in self.buffers[bin_index]:
                writer.write(record)
        print(f"Written {len(self.buffers[bin_index])} records to {output_file}")
        self.buffers[bin_index] = []
        self.buffer_counts[bin_index] = 0
        self.file_counts[bin_index] += 1

    def shuffle_and_write(self):
        for tfrecord_file in self.tfrecord_files:
            print(f"Processing file: {tfrecord_file}")
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            for raw_record in raw_dataset:
                bin_index = random.randint(0, self.num_bins - 1)
                self.buffers[bin_index].append(raw_record.numpy())
                self.buffer_counts[bin_index] += 1
                if self.buffer_counts[bin_index] >= self.batch_size:
                    print(f"Bin {bin_index} reached batch size. Writing to file.")
                    self._write_buffer_to_file(bin_index)
            print(f"After reading {tfrecord_file}:")
            for i in range(self.num_bins):
                print(f"  Bin {i} has {self.buffer_counts[i]} records")
        
        # Write remaining records in buffers
        for bin_index in range(self.num_bins):
            if self.buffer_counts[bin_index] > 0:
                print(f"Writing remaining records in bin {bin_index} to file.")
                self._write_buffer_to_file(bin_index)

# Example usage
data_dir = './data/tfrecords/fivetwelve_DO_NOT_CHANGE/train'
output_dir = './embedded_datasets/SEGS_SHUFFLED'
data_loader = DataLoader_segs(data_dir=data_dir, output_dir=output_dir)
data_loader.shuffle_and_write()