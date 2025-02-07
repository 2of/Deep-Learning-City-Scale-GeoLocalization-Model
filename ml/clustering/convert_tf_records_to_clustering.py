import tensorflow as tf
import os

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

def read_and_write_tfrecords(input_dir, output_filename):
    writer = tf.io.TFRecordWriter(output_filename)
    
    for tfrecord_filename in os.listdir(input_dir):
        if tfrecord_filename.endswith('.tfrecord'):
            print(f"Processing file: {tfrecord_filename}")
            raw_dataset = tf.data.TFRecordDataset(os.path.join(input_dir, tfrecord_filename))
            parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
            
            for parsed_record in parsed_dataset:
                feature = {
                    'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_record['id'].numpy()])),
                    'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[parsed_record['latitude'].numpy()])),
                    'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[parsed_record['longitude'].numpy()])),
                    'num_detections': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_record['num_detections'].numpy()])),
                    'color_histograms': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_record['color_histograms'].values.numpy())),
                    'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=parsed_record['text_embeddings'].values.numpy())),
                    'stacked_class_names_vector': tf.train.Feature(int64_list=tf.train.Int64List(value=parsed_record['stacked_class_names_vector'].values.numpy())),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    
    writer.close()
    print(f"All records have been written to {output_filename}")

if __name__ == "__main__":
    input_directory = "./MAIN_BRANCH_SHUFFLED/val/"
    output_tfrecord_filename = "./MAIN_BRANCH_SHUFFLED/all_val.tfrecord"
    read_and_write_tfrecords(input_directory, output_tfrecord_filename)