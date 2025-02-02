import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from train.models import fullimageModel  # Import the custom model
from train.OVERALL_utils import DataLoader
# Path to the TFRecord file and the H5 model
TFRECORD_FILE = './embedded_datasets/UNSEEN/predictions.tfrecord'
MODEL_FILE = './path_to_save_model/my_model3_test2.h5'  # Ensure this path is correct

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
        'stacked_confidences': tf.io.VarLenFeature(tf.float32),
    }
    return tf.io.parse_single_example(example, feature_description)

def read_tfrecord(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    return parsed_dataset

def predict_lat_lon(model, record):
    # Prepare the input data for the model
    input_data = {
        'text_embeddings': tf.sparse.to_dense(record['text_embeddings']),
        'color_histograms': tf.sparse.to_dense(record['color_histograms']),
        'class_names_vectors': tf.sparse.to_dense(record['stacked_class_names_vector']),
        'bboxes': tf.sparse.to_dense(record['stacked_bboxes'])
    }

    # Make predictions using the model
    predictions = model(input_data)
    predicted_latitude = predictions[0][0]
    predicted_longitude = predictions[0][1]
    return predicted_latitude, predicted_longitude

def main():
    # Load the H5 model within the custom object scope
    with custom_object_scope({'fullimageModel': fullimageModel}):
        model = tf.keras.models.load_model(MODEL_FILE)

    dataset = read_tfrecord(TFRECORD_FILE)
    print(f"Dataset size: {len(list(dataset))}")

    actual_latitudes = []
    actual_longitudes = []
    predicted_latitudes = []
    predicted_longitudes = []

    for record in dataset:
        print("Record sizes:")
        print(f"Latitude: {record['latitude'].shape}")
        print(f"Longitude: {record['longitude'].shape}")
        print(f"Text Embeddings: {record['text_embeddings'].dense_shape}")
        print(f"Color Histograms: {record['color_histograms'].dense_shape}")
        print(f"Class Names Vector: {record['stacked_class_names_vector'].dense_shape}")
        print(f"Bboxes: {record['stacked_bboxes'].dense_shape}")

        actual_latitude = record['latitude'].numpy()
        actual_longitude = record['longitude'].numpy()
        
        predicted_latitude, predicted_longitude = predict_lat_lon(model, record)
        
        actual_latitudes.append(actual_latitude)
        actual_longitudes.append(actual_longitude)
        predicted_latitudes.append(predicted_latitude)
        predicted_longitudes.append(predicted_longitude)
        
        print(f"Actual Latitude: {actual_latitude}, Actual Longitude: {actual_longitude}")
        print(f"Predicted Latitude: {predicted_latitude}, Predicted Longitude: {predicted_longitude}")
def predict(dataloader,model):
    latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = dataloader.get_next_batch()
           
    with custom_object_scope({'fullimageModel': fullimageModel}):
        model = tf.keras.models.load_model(MODEL_FILE)

    inputs = {
                'text_embeddings': text_embeddings,
                'color_histograms': color_histograms,
                'class_names_vectors': class_names_vectors,
                'bboxes': bboxes,
                'confidences': confidences
            }
    labels = tf.stack([latitudes, longitudes], axis=1)

    predictions = model.predict(inputs)
if __name__ == "__main__":
    
    
    
    test = DataLoader(data_dir="./embedded_datasets/UNSEEN", batch_size=512)

    
    predict(test,MODEL_FILE)

# from train.models import fullimageModel

# # Instantiate the model
# import tensorflow as tf
# from train.models import fullimageModel

# # Instantiate the model
# model = fullimageModel()

# # Build the model by providing an input shape
# input_shape = {
#     'text_embeddings': tf.TensorShape([None, 128]),
#     'color_histograms': tf.TensorShape([None, 128]),
#     'class_names_vectors': tf.TensorShape([None, 11]),
#     'bboxes': tf.TensorShape([None, 4])
# }
# model.build(input_shape)

# # Print the model summary to verify the architecture
# model.summary()

# # Save the model
# MODEL_FILE = './path_to_save_model/my_model3_NOOSS.h5'
# model.save(MODEL_FILE)

# import tensorflow as tf
# from train.models import fullimageModel

# # Instantiate the model
# model = fullimageModel()

# Build the model by providing an input shape
# input_shape = {
#     'text_embeddings': tf.TensorShape([None, 12, 128]),
#     'color_histograms': tf.TensorShape([None, 3, 256]),
#     'class_names_vectors': tf.TensorShape([None, 13]),
#     'bboxes': tf.TensorShape([None, 4])
# }
# model.build(input_shape)

# # Save the model
# MODEL_FILE = './path_to_save_model/my_model3_test2.h5'
# model.save(MODEL_FILE)


# import tensorflow as tf
# from tensorflow.keras.utils import custom_object_scope
# from train.models import fullimageModel

# # Define the input tensors based on the provided shapes
# example_input = {
#     'text_embeddings': tf.random.uniform([1, 12, 128]),  # B × 12 × 128
#     'color_histograms': tf.random.uniform([1, 3, 256]),  # B × 3 × 256
#     'class_names_vectors': tf.random.uniform([1, 13], maxval=8, dtype=tf.int32),  # B × N × 13
#     'bboxes': tf.random.uniform([1, 4])  # B × N × 4
# }

# # Load the model within the custom object scope
# MODEL_FILE = './path_to_save_model/my_model3_test2.h5'
# with custom_object_scope({'fullimageModel': fullimageModel}):
#     model = tf.keras.models.load_model(MODEL_FILE)

# # Make predictions
# predictions = model(example_input)
# print(predictions)
