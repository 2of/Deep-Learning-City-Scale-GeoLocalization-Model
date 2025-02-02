import os
import tensorflow as tf

def _parse_tf_example(example_proto):
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

def parse_tfrecord(example_proto):
    features = {
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'text_embeddings': tf.io.VarLenFeature(tf.float32),
        'class_names': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'color_histograms': tf.io.VarLenFeature(tf.float32)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_features['text_embeddings'] = tf.sparse.to_dense(parsed_features['color_histograms'])
    parsed_features['color_histograms'] = tf.sparse.to_dense(parsed_features['colour_histmgra'])
    return parsed_features

def load_and_filter_tfrecords(directory):
    tfrecord_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tfrecord')]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    filtered_dataset = raw_dataset.map(parse_tfrecord)
    return filtered_dataset

def save_filtered_tfrecords(filtered_dataset, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    for record in filtered_dataset:
        example = tf.train.Example(features=tf.train.Features(feature={
            'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[record['latitude'].numpy()])),
            'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[record['longitude'].numpy()])),
            'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=record['text_embeddings'].numpy())),
            'class_names': tf.train.Feature(bytes_list=tf.train.BytesList(value=[record['class_names'].numpy()])),
            'color_histograms': tf.train.Feature(float_list=tf.train.FloatList(value=record['color_histograms'].numpy()))
        }))
        writer.write(example.SerializeToString())
    writer.close()

directory = './embedded_datasets/MAIN_BRANCH_SHUFFLED/train'
output_file = './embedded_datasets/smaller/records.tfrecord'
filtered_dataset = load_and_filter_tfrecords(directory)
save_filtered_tfrecords(filtered_dataset, output_file)