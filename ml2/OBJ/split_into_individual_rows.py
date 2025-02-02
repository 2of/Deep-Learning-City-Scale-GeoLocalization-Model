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

    def _create_tf_example(self, id, latitude, longitude, text_embeddings, color_histogram):
        feature = {
            'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
            'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[latitude])),
            'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[longitude])),
            'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=text_embeddings.flatten().tolist())),
            'color_histogram': tf.train.Feature(float_list=tf.train.FloatList(value=color_histogram.flatten().tolist())),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _write_buffer_to_file(self, bin_index):
        output_file = os.path.join(self.output_dir, f'shuffled_{bin_index}_{self.file_counts[bin_index]}.tfrecord')
        with tf.io.TFRecordWriter(output_file) as writer:
            for record in self.buffers[bin_index]:
                writer.write(record.SerializeToString())
        print(f"Written {len(self.buffers[bin_index])} records to {output_file}")
        self.buffers[bin_index] = []
        self.buffer_counts[bin_index] = 0
        self.file_counts[bin_index] += 1

    def shuffle_and_write(self):
        for tfrecord_file in self.tfrecord_files:
            print(f"Processing file: {tfrecord_file}")
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            i = 0
            for raw_record in raw_dataset:
                parsed_record = self._parse_tf_example(raw_record)
                id = parsed_record['id'].numpy()
                latitude = parsed_record['latitude'].numpy()
                longitude = parsed_record['longitude'].numpy()
                text = parsed_record['text'].values.numpy()
                text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
                color_histograms = tf.sparse.to_dense(parsed_record['color_histograms']).numpy()
                num_detections = parsed_record['num_detections'].numpy()
               # print(i)
                i += 1
                
                
               # print(text_embeddings.shape)
                test_text_shape = text_embeddings.reshape(num_detections, 12, 128)
               # print(test_text_shape.shape)
                test_color_h = color_histograms.reshape(num_detections, 256)
               # print(id, latitude, longitude)
                
                for i in range(num_detections):
                #    print(test_text_shape)
                    tf_example = self._create_tf_example(id, latitude, longitude, test_text_shape[i], test_color_h[i])
                 #   print(tf_example)
                    
                    bin_index = random.randint(0, self.num_bins - 1)
                    self.buffers[bin_index].append(tf_example)
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
output_dir = './embedded_datasets/SEGS_SINGLE_PER_ROW/small'
data_loader = DataLoader_segs(data_dir=data_dir, output_dir=output_dir)
data_loader.shuffle_and_write()