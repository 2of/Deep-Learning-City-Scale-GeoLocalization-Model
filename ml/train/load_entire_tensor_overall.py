import tensorflow as tf

def parse_tf_example(example_proto):
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

def get_tf_record(tfrecord_file, batch_size=512):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tf_example)

    labels_list = []
    text_embeddings_list = []
    color_histograms_list = []
    num_detections_list = []
    stacked_class_names_vectors_list = []
    stacked_bboxes_list = []
    stacked_confidences_list = []

    for parsed_record in parsed_dataset:
        
        latitude = parsed_record['latitude'].numpy()
        longitude = parsed_record['longitude'].numpy()
        text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
        color_histograms = tf.sparse.to_dense(parsed_record['color_histograms']).numpy()
        num_detections = parsed_record['num_detections'].numpy()
        stacked_class_names_vector = parsed_record['stacked_class_names_vector']
        stacked_bboxes = tf.sparse.to_dense(parsed_record['stacked_bboxes']).numpy()
        stacked_confidences = tf.sparse.to_dense(parsed_record['stacked_confidences']).numpy()
        print(color_histograms.shape, "BEFOREHAND")
        # Convert sparse tensor to dense tensor
        stacked_class_names_vector = tf.sparse.to_dense(stacked_class_names_vector).numpy()

        # Ensure text embeddings are reshaped to match num_detections
        text_embeddings = text_embeddings.reshape((12, 128))

        # Ensure color histograms are reshaped to match num_detections
        color_histograms = color_histograms.reshape((3, 256))
        print(color_histograms.shape)
        # Add text embeddings with labels
        for embedding in text_embeddings:
            text_embeddings_list.append(embedding)
            labels_list.append((latitude, longitude))

        # Add color histograms with labels ( theres only one usually but one or two records have 2 text embeddings so it's just easier to do this)
  

        num_detections_list.append(num_detections)
        stacked_class_names_vectors_list.append(stacked_class_names_vector)
        stacked_bboxes_list.append(stacked_bboxes)
        stacked_confidences_list.append(stacked_confidences)

    # Convert lists to tensors
    labels_tensor = tf.convert_to_tensor(labels_list, dtype=tf.float32)
    text_embeddings_tensor = tf.convert_to_tensor(text_embeddings_list, dtype=tf.float32)
    color_histograms_tensor = tf.convert_to_tensor(color_histograms, dtype=tf.float32)
    print(color_histograms_tensor.shape, "AFTER")
    num_detections_tensor = tf.convert_to_tensor(num_detections_list, dtype=tf.int64)
    stacked_class_names_vectors_tensor = tf.convert_to_tensor(stacked_class_names_vectors_list, dtype=tf.int64)
    stacked_bboxes_tensor = tf.convert_to_tensor(stacked_bboxes_list, dtype=tf.float32)
    stacked_confidences_tensor = tf.convert_to_tensor(stacked_confidences_list, dtype=tf.float32)

    # Ensure the tensors are of shape BATCH_SIZE * whatever
    labels_tensor = labels_tensor[:batch_size]
    text_embeddings_tensor = text_embeddings_tensor[:batch_size]
    color_histograms_tensor = color_histograms_tensor[:batch_size]
    print("NOW", color_histograms_tensor.shape)
    num_detections_tensor = num_detections_tensor[:batch_size]
    stacked_class_names_vectors_tensor = stacked_class_names_vectors_tensor[:batch_size]
    stacked_bboxes_tensor = stacked_bboxes_tensor[:batch_size]
    stacked_confidences_tensor = stacked_confidences_tensor[:batch_size]

    return (labels_tensor, text_embeddings_tensor, color_histograms_tensor,
            num_detections_tensor, stacked_class_names_vectors_tensor,
            stacked_bboxes_tensor, stacked_confidences_tensor)

if __name__ == "__main__":
    tfrecord_file = './data/unseen_images/predictions21_SMALL.tfrecord'
    (labels_tensor, text_embeddings_tensor, color_histograms_tensor,
     num_detections_tensor, stacked_class_names_vectors_tensor,
     stacked_bboxes_tensor, stacked_confidences_tensor) = get_tf_record(tfrecord_file)

    print(f"Labels Tensor Shape: {labels_tensor.shape}")
    print(f"Text Embeddings Tensor Shape: {text_embeddings_tensor.shape}")
    print(f"Color Histograms Tensor Shape: {color_histograms_tensor.shape}")
    print(f"Num Detections Tensor Shape: {num_detections_tensor.shape}")
    print(f"Stacked Class Names Vectors Tensor Shape: {stacked_class_names_vectors_tensor.shape}")
    print(f"Stacked Bounding Boxes Tensor Shape: {stacked_bboxes_tensor.shape}")
    print(f"Stacked Confidences Tensor Shape: {stacked_confidences_tensor.shape}")

    # Print the first 15 values of labels
    print("First 15 values of labels:")
    print(labels_tensor[:15])

    # Print the first 3 tensor records in an understandable way
    for i in range(min(3, len(labels_tensor))):
        print(f"\nExample of tensor record {i + 1}:")
        print(f"ID: {i}")
        print(stacked_class_names_vectors_tensor[i])
        print(f"Latitude: {labels_tensor[i][0].numpy()}")
        print(f"Longitude: {labels_tensor[i][1].numpy()}")
        print(f"Number of Detections: {num_detections_tensor[i].numpy()}")
        print(f"Text Embeddings Shape: {text_embeddings_tensor[i * num_detections_tensor[i]:(i + 1) * num_detections_tensor[i]].shape}")
        print(f"Color Histogram Shape: {color_histograms_tensor[i * num_detections_tensor[i]:(i + 1) * num_detections_tensor[i]].shape}")
        print(color_histograms_tensor.shape)
        print(f"Stacked Class Names Vector Shape: {stacked_class_names_vectors_tensor[i].shape}")
        print(f"Stacked Bounding Boxes Shape: {stacked_bboxes_tensor[i].shape}")
        print(f"Stacked Confidence Scores Shape: {stacked_confidences_tensor[i].shape}")

    # Print the overall number of lines loaded
    print(f"Total number of lines loaded: {len(labels_tensor)}")



