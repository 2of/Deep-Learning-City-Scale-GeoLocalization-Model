import tensorflow as tf

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

def load_and_display_tfrecord(tfrecord_filename):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    
    # Initialize counters
    total_num_detections = 0
    record_count = 0
    non_zero_detection_count = 0  # Counter for records with num_detections > 0
    
    # Iterate over each parsed record and display the first five records
    for i, parsed_record in enumerate(parsed_dataset):
        # Accumulate the sum of 'num_detections'
        num_detections = parsed_record['num_detections'].numpy()
        total_num_detections += num_detections
        
        # Count records where num_detections is greater than 0
        if num_detections > 0:
            non_zero_detection_count += 1
        
        if i < 5:
            print(f"Record {i+1}:")
            print("ID:", parsed_record['id'].numpy())
            print("Latitude:", parsed_record['latitude'].numpy())
            print("Longitude:", parsed_record['longitude'].numpy())
            print("Number of Detections:", num_detections)
            # print("Color Histograms:", parsed_record['color_histograms'].values.numpy())
            print("Text Embeddings:", parsed_record['text_embeddings'].values.numpy())
            print("Stacked Class Names Vector:", parsed_record['stacked_class_names_vector'].values.numpy())
            print("\n")
        
        record_count += 1
    
    # Print the total number of records, sum of num_detections, and count of non-zero detections
    print(f"Total number of records in the TFRecord file: {record_count}")
    print(f"Total number of detections in all records: {total_num_detections}")
    print(f"Number of images with detections (num_detections > 0): {non_zero_detection_count}")

if __name__ == "__main__":
    tfrecord_filename = "./MAIN_BRANCH_SHUFFLED/mega_file4.tfrecord"
    load_and_display_tfrecord(tfrecord_filename)