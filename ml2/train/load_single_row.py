import tensorflow as tf

def parse_tf_example(example_proto):
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'text_embeddings': tf.io.VarLenFeature(tf.float32),
        'color_histogram': tf.io.VarLenFeature(tf.float32)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def get_tf_record(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tf_example)

    labels_list = []
    text_embeddings_list = []
    color_histograms_list = []

    for parsed_record in parsed_dataset:
        latitude = parsed_record['latitude'].numpy()
        longitude = parsed_record['longitude'].numpy()
        
        text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
        text_embeddings = tf.reshape(text_embeddings, [12, 128])
        print(text_embeddings.shape)
        color_histograms = tf.sparse.to_dense(parsed_record['color_histogram']).numpy()

        labels_list.append((latitude, longitude))
        text_embeddings_list.append(text_embeddings)
        color_histograms_list.append(color_histograms)

    # Convert lists to tensors
    labels_tensor = tf.convert_to_tensor(labels_list, dtype=tf.float32)
    text_embeddings_tensor = tf.convert_to_tensor(text_embeddings_list, dtype=tf.float32)
    color_histograms_tensor = tf.convert_to_tensor(color_histograms_list, dtype=tf.float32)

    return labels_tensor, text_embeddings_tensor, color_histograms_tensor

def get_tf_record_gen(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tf_example)

    for parsed_record in parsed_dataset:
        latitude = parsed_record['latitude'].numpy()
        longitude = parsed_record['longitude'].numpy()
        text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy().reshape((12, 128))
        color_histograms = tf.sparse.to_dense(parsed_record['color_histogram']).numpy().reshape((3, 256))

        yield latitude, longitude, text_embeddings, color_histograms

if __name__ == "__main__":
    tfrecord_file = './embedded_datasets/SEGS_SINGLE_PER_ROW/small/shuffled_1_0.tfrecord'
    labels_tensor, text_embeddings_tensor, color_histograms_tensor = get_tf_record(tfrecord_file)

    print(f"Labels Tensor Shape: {labels_tensor.shape}")
    print(f"Text Embeddings Tensor Shape: {text_embeddings_tensor.shape}")
    print(f"Color Histograms Tensor Shape: {color_histograms_tensor.shape}")

    # Print the first 15 values of labelsf
    print("First 15 values of labels:")
    print(labels_tensor[:15])

    # Print the first 3 tensor records in an understandable way
    for i in range(min(3, len(labels_tensor))):
        print(f"\nExample of tensor record {i + 1}:")
        print(f"Latitude: {labels_tensor[i][0].numpy()}")
        print(f"Longitude: {labels_tensor[i][1].numpy()}")
        print(f"Text Embeddings Shape: {text_embeddings_tensor[i].shape}")
        print(f"Color Histogram Shape: {color_histograms_tensor[i].shape}")

    # Print the overall number of lines loaded
    print(f"Total number of lines loaded: {len(labels_tensor)}")