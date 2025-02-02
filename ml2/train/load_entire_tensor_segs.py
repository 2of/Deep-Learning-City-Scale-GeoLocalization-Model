import tensorflow as tf

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

def get_tf_record(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tf_example)

    labels_list = []
    text_embeddings_list = []
    color_histograms_list = []
    num_detections_list = []

    for parsed_record in parsed_dataset:
        latitude = parsed_record['latitude'].numpy()
        longitude = parsed_record['longitude'].numpy()
        text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
        color_histograms = tf.sparse.to_dense(parsed_record['color_histograms']).numpy()
        num_detections = parsed_record['num_detections'].numpy()

        # Ensure text embeddings are reshaped to match num_detections
        text_embeddings = text_embeddings.reshape((num_detections, 12, 128))

        # Ensure color histograms are reshaped to match num_detections
        color_histograms = color_histograms.reshape((num_detections, 256))

        # Add text embeddings with labels
        for embedding in text_embeddings:
            text_embeddings_list.append(embedding)
            labels_list.append((latitude, longitude))

        # Add color histograms with labels
        for histogram in color_histograms:
            color_histograms_list.append(histogram)

        num_detections_list.append(num_detections)

    # Convert lists to tensors
    labels_tensor = tf.convert_to_tensor(labels_list, dtype=tf.float32)
    text_embeddings_tensor = tf.convert_to_tensor(text_embeddings_list, dtype=tf.float32)
    color_histograms_tensor = tf.convert_to_tensor(color_histograms_list, dtype=tf.float32)
    num_detections_tensor = tf.convert_to_tensor(num_detections_list, dtype=tf.int64)

    return labels_tensor, text_embeddings_tensor, color_histograms_tensor, num_detections_tensor


def get_tf_record_gen(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tf_example)

    for parsed_record in parsed_dataset:
        latitude = parsed_record['latitude'].numpy()
        longitude = parsed_record['longitude'].numpy()
        text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
        color_histograms = tf.sparse.to_dense(parsed_record['color_histograms']).numpy()
        num_detections = parsed_record['num_detections'].numpy()

        # Ensure text embeddings are reshaped to match num_detections
        text_embeddings = text_embeddings.reshape((num_detections, 12, 128))

        # Ensure color histograms are reshaped to match num_detections
        color_histograms = color_histograms.reshape((num_detections, 256))

        for i in range(num_detections):
            yield latitude, longitude, text_embeddings[i], color_histograms[i]
if __name__ == "__main__":
    tfrecord_file = './data/tfrecords/fivetwelve_DO_NOT_CHANGE/test/batch_1.tfrecord'
    labels_tensor, text_embeddings_tensor, color_histograms_tensor, num_detections_tensor = get_tf_record(tfrecord_file)

    print(f"Labels Tensor Shape: {labels_tensor.shape}")
    print(f"Text Embeddings Tensor Shape: {text_embeddings_tensor.shape}")
    print(f"Color Histograms Tensor Shape: {color_histograms_tensor.shape}")
    print(f"Num Detections Tensor Shape: {num_detections_tensor.shape}")

    # Print the first 15 values of labels
    print("First 15 values of labels:")
    print(labels_tensor[:15])

    # Print the first 3 tensor records in an understandable way
    for i in range(min(3, len(labels_tensor))):
        print(f"\nExample of tensor record {i + 1}:")
        print(f"Latitude: {labels_tensor[i][0].numpy()}")
        print(f"Longitude: {labels_tensor[i][1].numpy()}")
        print(f"Number of Detections: {num_detections_tensor[i].numpy()}")
        print(f"Text Embeddings Shape: {text_embeddings_tensor[i * num_detections_tensor[i]:(i + 1) * num_detections_tensor[i]].shape}")
        print(f"Color Histogram Shape: {color_histograms_tensor[i * num_detections_tensor[i]:(i + 1) * num_detections_tensor[i]].shape}")

    # Print the overall number of lines loaded
    print(f"Total number of lines loaded: {len(labels_tensor)}")