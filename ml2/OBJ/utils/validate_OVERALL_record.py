import tensorflow as tf

def parse_tfrecord_fn(example):
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
    return tf.io.parse_single_example(example, feature_description)

def read_tfrecord(tfrecord_filename):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    
    # Initialize lists to store each tensor for all records
    ids = []
    latitudes = []
    longitudes = []
    texts = []
    text_embeddings = []
    color_histograms = []
    num_detections = []
    stacked_class_names_vectors = []
    stacked_bboxes = []
    stacked_confidences = []

    # Iterate over each parsed record and append the tensors to the respective lists
    for i, parsed_record in enumerate(parsed_dataset):
        ids.append(parsed_record['id'].numpy())
        latitudes.append(parsed_record['latitude'].numpy())
        longitudes.append(parsed_record['longitude'].numpy())
        texts.append([t.numpy().decode('utf-8') for t in parsed_record['text'].values])
        text_embeddings.append(parsed_record['text_embeddings'].values.numpy())
        color_histograms.append(parsed_record['color_histograms'].values.numpy())
        num_detections.append(parsed_record['num_detections'].numpy())
        stacked_class_names_vectors.append(parsed_record['stacked_class_names_vector'].values.numpy())
        stacked_bboxes.append(parsed_record['stacked_bboxes'].values.numpy())
        stacked_confidences.append(parsed_record['stacked_confidences'].values.numpy())

        # Print the first 2 records
        if i < 5:
            print(f"Record {i+1}:")
            print("ID:", ids[-1])
            print("Latitude:", latitudes[-1])
            print("Longitude:", longitudes[-1])
            print("Text:", texts[-1])
            print("Text Embeddings:", text_embeddings[-1])
            # print("Color Histograms:", color_histograms[-1])
            print("Number of Detections:", num_detections[-1])
            print("Stacked Class Names Vector:", stacked_class_names_vectors[-1])
            print("Stacked Bounding Boxes:", stacked_bboxes[-1])
            print("Stacked Confidence Scores:", stacked_confidences[-1])
            print("\n")

    # Print the number of records
    print(f"Number of records in the TFRecord file: {len(ids)}")

if __name__ == "__main__":
    tfrecord_filename = "./data/tfrecords/lilMainBranc/output_0.tfrecord"
    read_tfrecord(tfrecord_filename)