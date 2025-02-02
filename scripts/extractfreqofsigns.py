import tensorflow as tf
import os
import pandas as pd

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

# Function to extract latitude and longitude pairs from all TFRecord files in a directory
def extract_lat_lon_pairs(tfrecord_dir):
    lat_lon_counts = {}
    tfrecord_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord')]
    print(f"Found {len(tfrecord_files)} TFRecord files in {tfrecord_dir}")

    total_detections = 0
    max_detections = 0
    min_detections = float('inf')

    for tfrecord_file in tfrecord_files:
        print(f"Processing TFRecord file: {tfrecord_file}")
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        parsed_dataset = raw_dataset.map(parse_tf_example)
        for parsed_record in parsed_dataset:
            latitude = parsed_record['latitude'].numpy()
            longitude = parsed_record['longitude'].numpy()
            num_detections = parsed_record['num_detections'].numpy()
            lat_lon_pair = (latitude, longitude)
            if lat_lon_pair in lat_lon_counts:
                lat_lon_counts[lat_lon_pair] += num_detections
            else:
                lat_lon_counts[lat_lon_pair] = num_detections

            total_detections += num_detections
            if num_detections > max_detections:
                max_detections = num_detections
            if num_detections < min_detections:
                min_detections = num_detections

    print(f"Total unique latitude/longitude pairs: {len(lat_lon_counts)}")
    print(f"Total number of detections: {total_detections}")
    print(f"Maximum number of detections for any pair: {max_detections}")
    print(f"Minimum number of detections for any pair: {min_detections}")
    return lat_lon_counts, total_detections, max_detections, min_detections

if __name__ == "__main__":
    tfrecord_dir = "./data/tfrecords/fivetwelve"
    output_csv = "./lat_lon_counts.csv"
    
    print("Starting to extract latitude/longitude pairs...")
    lat_lon_counts, total_detections, max_detections, min_detections = extract_lat_lon_pairs(tfrecord_dir)
    
    # Convert the dictionary to a DataFrame and save to CSV
    lat_lon_counts_df = pd.DataFrame(list(lat_lon_counts.items()), columns=['lat_lon_pair', 'count'])
    lat_lon_counts_df.to_csv(output_csv, index=False)
    print(f"Latitude/Longitude counts written to {output_csv}")
    print("Finished extracting latitude/longitude pairs.")
    print(f"Total number of detections: {total_detections}")
    print(f"Maximum number of detections for any pair: {max_detections}")
    print(f"Minimum number of detections for any pair: {min_detections}")