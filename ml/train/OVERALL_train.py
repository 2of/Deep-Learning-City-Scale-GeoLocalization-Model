import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping
from models import fullimageModel
from OVERALL_utils import DataLoader
import numpy as np
import csv
from geopy.distance import geodesic
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.spatial.distance import cosine

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize dictionaries to store MSE values
test_mses = {}
val_mses = {}
train_mses = {}

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: loss = {logs['loss']}, mse = {logs['mse']}")

def randomize_endpoints(latitudes, longitudes, candidate_area_lat, candidate_area_lon, epoch, total_epochs):
    # Gradually increase randomization over epochs
    factor = epoch / total_epochs
    latitudes += np.random.uniform(-candidate_area_lat * factor, candidate_area_lat * factor, size=latitudes.shape)
    longitudes += np.random.uniform(-candidate_area_lon * factor, candidate_area_lon * factor, size=longitudes.shape)
    
    # Clip to ensure values stay within the bounding box
    latitudes = np.clip(latitudes, bounding_box['min_lat'], bounding_box['max_lat'])
    longitudes = np.clip(longitudes, bounding_box['min_lon'], bounding_box['max_lon'])
    
    return latitudes, longitudes

def normalize_coordinates(latitudes, longitudes, bounding_box):
    latitudes = (latitudes - bounding_box['min_lat']) / (bounding_box['max_lat'] - bounding_box['min_lat'])
    longitudes = (longitudes - bounding_box['min_lon']) / (bounding_box['max_lon'] - bounding_box['min_lon'])
    return latitudes, longitudes

def denormalize_coordinates(latitudes, longitudes, bounding_box):
    latitudes = latitudes * (bounding_box['max_lat'] - bounding_box['min_lat']) + bounding_box['min_lat']
    longitudes = longitudes * (bounding_box['max_lon'] - bounding_box['min_lon']) + bounding_box['min_lon']
    return latitudes, longitudes

def adjust_coordinates(latitudes, longitudes):
    # latitudes -= 41.9
    # longitudes += 87.6
    return latitudes, longitudes

def latitude_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true[:, 0], y_pred[:, 0])

def longitude_loss(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true[:, 1], y_pred[:, 1])

def combined_loss(y_true, y_pred):
    lat_loss = latitude_loss(y_true, y_pred)
    lon_loss = longitude_loss(y_true, y_pred)
    return lat_loss + lon_loss  # You can adjust the weights if needed

def calculate_geodesic_distance(y_true, y_pred):
    distances = []
    for i in range(len(y_true)):
        true_coords = (y_true[i][0], y_true[i][1])
        pred_coords = (y_pred[i][0], y_pred[i][1])
        distances.append(geodesic(true_coords, pred_coords).meters)
    return np.mean(distances), np.median(distances)

def calculate_directional_error(y_true, y_pred):
    errors = []
    for i in range(len(y_true)):
        true_vector = [y_true[i][0], y_true[i][1]]
        pred_vector = [y_pred[i][0], y_pred[i][1]]
        errors.append(cosine(true_vector, pred_vector))
    return np.mean(errors)

def load_and_compile_model(model_class, learning_rate=0.0001):
    model = model_class()
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['mse'])
    return model

def train_model(train_dir, val_dir, test_dir, batch_size, epochs, candidate_area_lat, candidate_area_lon, filename, model):
    # Create a directory to store the results
    os.makedirs(f"RUN_{filename}", exist_ok=True)

    # Load the datasets using DataLoader
    print("Loading datasets...")
    train_loader = DataLoader(train_dir, batch_size, shuffle=True)
    val_loader = DataLoader(val_dir, batch_size, shuffle=False)
    test_loader = DataLoader(test_dir, batch_size, shuffle=False)
    print("Datasets loaded")

    # Set up the tensor tensor board TensorBoard callback
    log_dir = "./logs"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Learning rate scheduler with piecewise dropoff 
    def scheduler(epoch, lr):
        if epoch < 111:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    lr_scheduler = LearningRateScheduler(scheduler)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Open CSV files for writing MSEs and other metrics
    with open(f'RUN_{filename}/train_mses_OVERALL.csv', 'w', newline='') as train_file, \
         open(f'RUN_{filename}/val_mses_OVERALL.csv', 'w', newline='') as val_file, \
         open(f'RUN_{filename}/test_mses_OVERALL.csv', 'w', newline='') as test_file, \
         open(f'RUN_{filename}/train_mses_LAT.csv', 'w', newline='') as train_lat_file, \
         open(f'RUN_{filename}/val_mses_LAT.csv', 'w', newline='') as val_lat_file, \
         open(f'RUN_{filename}/test_mses_LAT.csv', 'w', newline='') as test_lat_file, \
         open(f'RUN_{filename}/train_mses_LONG.csv', 'w', newline='') as train_long_file, \
         open(f'RUN_{filename}/val_mses_LONG.csv', 'w', newline='') as val_long_file, \
         open(f'RUN_{filename}/test_mses_LONG.csv', 'w', newline='') as test_long_file, \
         open(f'RUN_{filename}/val_mae.csv', 'w', newline='') as val_mae_file, \
         open(f'RUN_{filename}/val_rmse.csv', 'w', newline='') as val_rmse_file, \
         open(f'RUN_{filename}/val_geodesic_distance.csv', 'w', newline='') as val_geodesic_distance_file, \
         open(f'RUN_{filename}/val_mean_geodesic_error.csv', 'w', newline='') as val_mean_geodesic_error_file, \
         open(f'RUN_{filename}/val_median_absolute_error.csv', 'w', newline='') as val_median_absolute_error_file, \
         open(f'RUN_{filename}/val_directional_error.csv', 'w', newline='') as val_directional_error_file:

        train_writer = csv.writer(train_file)
        val_writer = csv.writer(val_file)
        test_writer = csv.writer(test_file)
        train_lat_writer = csv.writer(train_lat_file)
        val_lat_writer = csv.writer(val_lat_file)
        test_lat_writer = csv.writer(test_lat_file)
        train_long_writer = csv.writer(train_long_file)
        val_long_writer = csv.writer(val_long_file)
        test_long_writer = csv.writer(test_long_file)
        val_mae_writer = csv.writer(val_mae_file)
        val_rmse_writer = csv.writer(val_rmse_file)
        val_geodesic_distance_writer = csv.writer(val_geodesic_distance_file)
        val_mean_geodesic_error_writer = csv.writer(val_mean_geodesic_error_file)
        val_median_absolute_error_writer = csv.writer(val_median_absolute_error_file)
        val_directional_error_writer = csv.writer(val_directional_error_file)

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Training
            epoch_train_mses = []
            epoch_train_lat_mses = []
            epoch_train_long_mses = []
            for batch in range(len(train_loader.tfrecord_files)):
                print(f"Training batch {batch + 1}/{len(train_loader.tfrecord_files)}")
                latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = train_loader.get_next_batch()
                
                # Adjust coordinates
                latitudes, longitudes = adjust_coordinates(latitudes, longitudes)
                # print(latitudes)
                # Normalize coordinates
                latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
                
                # Randomize the initial ending points within the candidate area after warm-up phase
                if epoch >= 220:
                    latitudes, longitudes = randomize_endpoints(latitudes, longitudes, candidate_area_lat, candidate_area_lon, epoch, epochs)
                inputs = {
                    'text_embeddings': text_embeddings,
                    'color_histograms': color_histograms,
                    'class_names_vectors': class_names_vectors,
                    'bboxes': bboxes,
                    'confidences': confidences
                    
                }
                
                
                print(inputs)
                labels = tf.stack([latitudes, longitudes], axis=1)
                loss, mse = model.train_on_batch(inputs, labels)
                lat_mse = latitude_loss(labels, model.predict(inputs)).numpy().mean()
                long_mse = longitude_loss(labels, model.predict(inputs)).numpy().mean()
                print(f"Batch {batch + 1} - Loss: {loss}, MSE: {mse}, MSE_LAT: {lat_mse}, MSE_LONG: {long_mse}")
                epoch_train_mses.append(mse)
                epoch_train_lat_mses.append(lat_mse)
                epoch_train_long_mses.append(long_mse)

            train_writer.writerow(epoch_train_mses)
            train_lat_writer.writerow(epoch_train_lat_mses)
            train_long_writer.writerow(epoch_train_long_mses)

            # Validation
            epoch_val_mses = []
            epoch_val_lat_mses = []
            epoch_val_long_mses = []
            epoch_val_mae = []
            epoch_val_rmse = []
            epoch_val_geodesic_distance = []
            epoch_val_mean_geodesic_error = []
            epoch_val_median_absolute_error = []
            epoch_val_directional_error = []
            val_loss, val_mse = 0, 0
            for batch in range(len(val_loader.tfrecord_files)):
                print(f"Validating batch {batch + 1}/{len(val_loader.tfrecord_files)}")
                latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = val_loader.get_next_batch()
                
                # Adjust coordinates
                latitudes, longitudes = adjust_coordinates(latitudes, longitudes)
                
                # Normalize coordinates
                latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
                
                inputs = {
                    'text_embeddings': text_embeddings,
                    'color_histograms': color_histograms,
                    'class_names_vectors': class_names_vectors,
                    'bboxes': bboxes,
                    'confidences': confidences
                }
                labels = tf.stack([latitudes, longitudes], axis=1)
                loss, mse = model.test_on_batch(inputs, labels)
                lat_mse = latitude_loss(labels, model.predict(inputs)).numpy().mean()
                long_mse = longitude_loss(labels, model.predict(inputs)).numpy().mean()
                val_loss += loss
                val_mse += mse
                
                # Denormalize coordinates for printing truth and prediction values
                predicted_lat_long = model.predict(inputs)
                predicted_lat_long_denorm = denormalize_coordinates(predicted_lat_long[:, 0], predicted_lat_long[:, 1], bounding_box)
                true_lat_long_denorm = denormalize_coordinates(labels[:, 0], labels[:, 1], bounding_box)

                # Calculate additional metrics
                mae = mean_absolute_error(true_lat_long_denorm, predicted_lat_long_denorm)
                rmse = np.sqrt(mean_squared_error(true_lat_long_denorm, predicted_lat_long_denorm))
                mean_geodesic_error, median_geodesic_error = calculate_geodesic_distance(true_lat_long_denorm, predicted_lat_long_denorm)
                directional_error = calculate_directional_error(true_lat_long_denorm, predicted_lat_long_denorm)

                print(f"Batch {batch + 1} - Validation Loss: {loss}, Validation MSE: {mse}, MSE_LAT: {lat_mse}, MSE_LONG: {long_mse}, MAE: {mae}, RMSE: {rmse}, Mean Geodesic Error: {mean_geodesic_error}, Median Geodesic Error: {median_geodesic_error}, Directional Error: {directional_error}")
                epoch_val_mses.append(mse)
                epoch_val_lat_mses.append(lat_mse)
                epoch_val_long_mses.append(long_mse)
                epoch_val_mae.append(mae)
                epoch_val_rmse.append(rmse)
                epoch_val_geodesic_distance.append(mean_geodesic_error)
                epoch_val_mean_geodesic_error.append(mean_geodesic_error)
                epoch_val_median_absolute_error.append(median_geodesic_error)
                epoch_val_directional_error.append(directional_error)

            val_loss /= len(val_loader.tfrecord_files)
            val_mse /= len(val_loader.tfrecord_files)
            print(f"Epoch {epoch + 1} - Validation loss: {val_loss}, Validation MSE: {val_mse}")

            val_writer.writerow(epoch_val_mses)
            val_lat_writer.writerow(epoch_val_lat_mses)
            val_long_writer.writerow(epoch_val_long_mses)
            val_mae_writer.writerow(epoch_val_mae)
            val_rmse_writer.writerow(epoch_val_rmse)
            val_geodesic_distance_writer.writerow(epoch_val_geodesic_distance)
            val_mean_geodesic_error_writer.writerow(epoch_val_mean_geodesic_error)
            val_median_absolute_error_writer.writerow(epoch_val_median_absolute_error)
            val_directional_error_writer.writerow(epoch_val_directional_error)

            # Custom callback
            CustomCallback().on_epoch_end(epoch, logs={'loss': val_loss, 'mse': val_mse})

        # Testing
        epoch_test_mses = []
        epoch_test_lat_mses = []
        epoch_test_long_mses = []
        test_loss, test_mse = 0, 0
        for batch in range(len(test_loader.tfrecord_files)):
            print(f"Testing batch {batch + 1}/{len(test_loader.tfrecord_files)}")
            latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = test_loader.get_next_batch()
            
            # Adjust coordinates
            latitudes, longitudes = adjust_coordinates(latitudes, longitudes)
            
            # Normalize coordinates
            latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
            
            inputs = {
                'text_embeddings': text_embeddings,
                'color_histograms': color_histograms,
                'class_names_vectors': class_names_vectors,
                'bboxes': bboxes,
                'confidences': confidences
            }
            labels = tf.stack([latitudes, longitudes], axis=1)
            loss, mse = model.test_on_batch(inputs, labels)
            lat_mse = latitude_loss(labels, model.predict(inputs)).numpy().mean()
            long_mse = longitude_loss(labels, model.predict(inputs)).numpy().mean()
            test_loss += loss
            test_mse += mse
            print(f"Batch {batch + 1} - Test Loss: {loss}, Test MSE: {mse}, MSE_LAT: {lat_mse}, MSE_LONG: {long_mse}")
            epoch_test_mses.append(mse)
            epoch_test_lat_mses.append(lat_mse)
            epoch_test_long_mses.append(long_mse)

        test_loss /= len(test_loader.tfrecord_files)
        test_mse /= len(test_loader.tfrecord_files)
        print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

        test_writer.writerow(epoch_test_mses)
        test_lat_writer.writerow(epoch_test_lat_mses)
        test_long_writer.writerow(epoch_test_long_mses)
    model.save(f'RUN_{filename}/my_model3.h5')
                


if __name__ == "__main__":
    train_dir = "./embedded_datasets/MAIN_BRANCH_SHUFFLED/train"
    val_dir = "./embedded_datasets/MAIN_BRANCH_SHUFFLED/val"
    test_dir = "./embedded_datasets/MAIN_BRANCH_SHUFFLED/test" # Adjust these ones for sanity check and development
    batch_size = 512  # Adjust as needed
    epochs = 50  # Adjust as needed
    full = load_and_compile_model(fullimageModel,0.00001)
    # Define the bounding box
    # Calculate the candidate area
    bounding_box = {'min_lat': 41.87689068301984, 'max_lat': 41.91464908273332, 'min_lon': -87.68513023853302, 'max_lon': -87.62606263160706}
    

    candidate_area_lat = (bounding_box['max_lat'] - bounding_box['min_lat']) / 2
    candidate_area_lon = (bounding_box['max_lon'] - bounding_box['min_lon']) / 2

    train_model(train_dir, val_dir, test_dir, batch_size, epochs, candidate_area_lat, candidate_area_lon,"TEST2", model=full)
