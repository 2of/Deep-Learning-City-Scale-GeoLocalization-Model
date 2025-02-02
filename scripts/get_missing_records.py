import tensorflow as tf
import pandas as pd
import os

''' Gets the missing records from the tfrecords and the pickle file; if exist in the pickle but not in the tfrecords, it will be written to a CSV file. '''

# Function to parse TFRecord example
def parse_tf_example(example_proto):
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

# Function to extract IDs from all TFRecord files in a directory
def extract_ids_from_tfrecords(tfrecord_dir):
    ids = set()
    tfrecord_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord')]
    print(f"Found {len(tfrecord_files)} TFRecord files in {tfrecord_dir}")

    for tfrecord_file in tfrecord_files:
        print(f"Extracting IDs from TFRecord file: {tfrecord_file}")
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        parsed_dataset = raw_dataset.map(parse_tf_example)
        for parsed_record in parsed_dataset:
            ids.add(parsed_record['id'].numpy())
        print(f"Extracted {len(ids)} IDs from TFRecord file: {tfrecord_file}")

    print(f"Total extracted IDs from all TFRecord files: {len(ids)}")
    return ids

# Function to extract IDs from pickle file
def extract_ids_from_pickle(pickle_file):
    print(f"Extracting IDs from pickle file: {pickle_file}")
    df = pd.read_pickle(pickle_file)
    ids = set(df['id'])
    print(f"Extracted {len(ids)} IDs from pickle file.")
    return ids

# Main function to find missing IDs
def find_missing_ids(tfrecord_dir, pickle_file):
    tfrecord_ids = extract_ids_from_tfrecords(tfrecord_dir)
    pickle_ids = extract_ids_from_pickle(pickle_file)
    missing_ids = pickle_ids - tfrecord_ids
    return missing_ids

if __name__ == "__main__":
    tfrecord_dir = "./data/tfrecords/fivetwelve"
    pickle_file = "./chicago.pkl"
    output_csv = "./missing_ids2.csv"
    
    print("Starting to find missing IDs...")
    missing_ids = find_missing_ids(tfrecord_dir, pickle_file)
    if missing_ids:
        print(f"Number of missing IDs: {len(missing_ids)}") 
        missing_ids_df = pd.DataFrame(list(missing_ids), columns=['missing_id'])
        missing_ids_df.to_csv(output_csv, index=False)
        print(f"Missing IDs written to {output_csv}")
    else:
        print("No IDs are missing.")
    print("Finished finding missing IDs.")