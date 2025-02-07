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

def display_tfrecord(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tf_example)

    row_count = 0
    unique_ids = set()
    unique_text_embeddings = set()
    unique_color_histograms = set()

    for parsed_record in parsed_dataset:
        id = parsed_record['id'].numpy()
        latitude = parsed_record['latitude'].numpy()
        longitude = parsed_record['longitude'].numpy()
        text = [t.numpy().decode('utf-8') for t in tf.sparse.to_dense(parsed_record['text'])]
        text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
        color_histograms = tf.sparse.to_dense(parsed_record['color_histograms']).numpy()
        num_detections = parsed_record['num_detections'].numpy()

        # print(f"ID: {id}")
        # print(f"Latitude: {latitude}")
        # print(f"Longitude: {longitude}")
        # print(f"Text: {text}")
        # print(f"Text Embeddings Size: {text_embeddings.size}")
        # print(f"Color Histograms Size: {color_histograms.size}")
        # print(f"Number of Detections: {num_detections}")

        # Ensure text embeddings are reshaped to match num_detections
        text_embeddings = text_embeddings.reshape((num_detections, 12, 128))
        # print(f"Text Embeddings Shape: {text_embeddings.shape}")

        # Determine the number of histograms
        histogram_size = len(color_histograms) // num_detections if num_detections > 0 else 0
        num_histograms = len(color_histograms) // histogram_size if histogram_size > 0 else 0
        # print(f"Number of histograms: {num_histograms}")

        # Ensure the number of histograms matches the number of detections
        # if num_histograms != num_detections:
            # print(f"Warning: Number of histograms ({num_histograms}) does not match number of detections ({num_detections})")

        # Unpack and display each histogram embedding
        # print("Color Histograms:")
        # for i in range(num_histograms):
        #     histogram = color_histograms[i * histogram_size:(i + 1) * histogram_size]
        #     print(f"  Histogram {i} Shape: {histogram.shape}")

        # print("\n")

        unique_ids.add(id)
        unique_text_embeddings.add(text_embeddings.tobytes())
        unique_color_histograms.add(color_histograms.tobytes())

        row_count += 1

    print(f"Total number of rows: {row_count}")
    print(f"Total number of unique images: {len(unique_ids)}")
    print(f"Total number of unique text embeddings: {len(unique_text_embeddings)}")
    print(f"Total number of unique color histograms: {len(unique_color_histograms)}")

if __name__ == "__main__":
    tfrecord_file = './data/combined.tfrecord'
    display_tfrecord(tfrecord_file)
    
    
    '''
    
    ID: 766887924011438
Latitude: 41.90553283691406
Longitude: -87.67291259765625


'''