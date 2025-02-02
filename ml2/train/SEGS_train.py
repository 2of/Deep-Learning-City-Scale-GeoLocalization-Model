import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping
from models import fullimageModel, concatModelForSegments_attn
from SEGS_utils import *
import numpy as np
import csv
from geopy.distance import geodesic
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.spatial.distance import cosine
import json
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_OVERRIDE = 112312 # Just put 1 here to only load one entry...... 
# Initialize dictionaries to store MSE values
test_mses = {}
val_mses = {}
train_mses = {}

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file, base_dir):
        super(CustomCallback, self).__init__()
        self.log_file = log_file
        self.base_dir = base_dir
        # Create the log file and write the header
        with open(self.log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss', 'MSE', 'Learning Rate', 'Layer', 'Weights Mean', 'Weights Std', 'Weights Min', 'Weights Max', 'Biases Mean', 'Biases Std', 'Biases Min', 'Biases Max'])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_dir = os.path.join(self.base_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        with open(self.log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for layer in self.model.layers:
                weights = layer.get_weights()
                if len(weights) > 0:
                    weights_mean = np.mean(weights[0])
                    weights_std = np.std(weights[0])
                    weights_min = np.min(weights[0])
                    weights_max = np.max(weights[0])
                    biases_mean = np.mean(weights[1]) if len(weights) > 1 else None
                    biases_std = np.std(weights[1]) if len(weights) > 1 else None
                    biases_min = np.min(weights[1]) if len(weights) > 1 else None
                    biases_max = np.max(weights[1]) if len(weights) > 1 else None
                    writer.writerow([
                        epoch,
                        logs.get('loss'),
                        logs.get('mse'),
                        float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
                        layer.name,
                        weights_mean,
                        weights_std,
                        weights_min,
                        weights_max,
                        biases_mean,
                        biases_std,
                        biases_min,
                        biases_max
                    ])
                    # Save histogram data to separate files
                    weights_hist, weights_bins = np.histogram(weights[0])
                    biases_hist, biases_bins = np.histogram(weights[1]) if len(weights) > 1 else (None, None)
                    
                    weights_hist_file = os.path.join(epoch_dir, f'{layer.name}_weights_histogram.json')
                    with open(weights_hist_file, 'w') as hist_file:
                        json.dump({'hist': weights_hist.tolist(), 'bins': weights_bins.tolist()}, hist_file, indent=4)
                    
                    if biases_hist is not None:
                        biases_hist_file = os.path.join(epoch_dir, f'{layer.name}_biases_histogram.json')
                        with open(biases_hist_file, 'w') as hist_file:
                            json.dump({'hist': biases_hist.tolist(), 'bins': biases_bins.tolist()}, hist_file, indent=4)
                    
def train_model(train_dir, val_dir, test_dir, batch_size, epochs, candidate_area_lat, candidate_area_lon,save_dir):
    # Create a directory to store the results
    run_id = np.random.randint(10000)
    os.makedirs(f"RUN_{run_id}", exist_ok=True)

    # Load the datasets using DataLoader
   
    # Initialize the model
    model = concatModelForSegments_attn()
    optimizer = Adam(learning_rate=0.0001)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['mse'])
    print("Model compiled")

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

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Custom callback for detailed logging
    custom_callback = CustomCallback(os.path.join(save_dir, 'detailed_logs.csv'),save_dir)

    # Combine all callbacks
    callbacks = [tensorboard_callback, lr_scheduler, early_stopping, custom_callback]

    # Set the model attribute for all callbacks
    for callback in callbacks:
        callback.set_model(model)
        
        
    # Open CSV files for writing MSEs and other metrics
    with open(os.path.join(save_dir, 'train_mses_OVERALL.csv'), 'w', newline='') as train_file, \
         open(os.path.join(save_dir, 'val_mses_OVERALL.csv'), 'w', newline='') as val_file, \
         open(os.path.join(save_dir, 'test_mses_OVERALL.csv'), 'w', newline='') as test_file, \
         open(os.path.join(save_dir, 'train_mses_LAT.csv'), 'w', newline='') as train_lat_file, \
         open(os.path.join(save_dir, 'val_mses_LAT.csv'), 'w', newline='') as val_lat_file, \
         open(os.path.join(save_dir, 'test_mses_LAT.csv'), 'w', newline='') as test_lat_file, \
         open(os.path.join(save_dir, 'train_mses_LONG.csv'), 'w', newline='') as train_long_file, \
         open(os.path.join(save_dir, 'val_mses_LONG.csv'), 'w', newline='') as val_long_file, \
         open(os.path.join(save_dir, 'test_mses_LONG.csv'), 'w', newline='') as test_long_file:
        
        train_writer = csv.writer(train_file)
        train_writer = csv.writer(train_file)
        val_writer = csv.writer(val_file)
        test_writer = csv.writer(test_file)
        train_lat_writer = csv.writer(train_lat_file)
        val_lat_writer = csv.writer(val_lat_file)
        test_lat_writer = csv.writer(test_lat_file)
        train_long_writer = csv.writer(train_long_file)
        val_long_writer = csv.writer(val_long_file)
        test_long_writer = csv.writer(test_long_file)


        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("Loading datasets...")
            train_loader = DataLoader(train_dir, batch_size, shuffle=True)
            val_loader = DataLoader(val_dir, batch_size, shuffle=False)
            test_loader = DataLoader(test_dir, batch_size, shuffle=False)
            print("Datasets loaded")


            # Training
            epoch_train_mses = []
            epoch_train_lat_mses = []
            epoch_train_long_mses = []
            print("FULL LENGTH is ", (len(train_loader.tfrecord_files)))
            for batch in range(min(BATCH_OVERRIDE,len(train_loader.tfrecord_files))):
                
                print(f"Training batch {batch + 1}/{len(train_loader.tfrecord_files)}")
                # latitudes, longitudes, text_embeddings, color_histograms, class_names_vectors, bboxes, confidences = train_loader.get_next_batch()
               
                stuff = train_loader.get_next_batch()
                if stuff is None:
                    continue
                latitudes,longitudes,text_embeddings, color_histograms  = stuff
    
                # Adjust coordinates
                latitudes, longitudes = adjust_coordinates(latitudes, longitudes)
                
                # Normalize coordinates
                latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
                
                # Randomize the initial ending points within the candidate area after warm-up phase
                if epoch >= 220:
                    latitudes, longitudes = randomize_endpoints(latitudes, longitudes, candidate_area_lat, candidate_area_lon, epoch, epochs)
                inputs = {
                    'text_embeddings': text_embeddings,
                    'color_histograms': color_histograms,
                    
                }
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
            for batch in range(min(BATCH_OVERRIDE,len(val_loader.tfrecord_files))):
                print(f"Validating batch {batch + 1}/{len(val_loader.tfrecord_files)}")

                
                stuff = val_loader.get_next_batch()
                if stuff is None:
                    continue
                latitudes,longitudes,text_embeddings, color_histograms  = stuff
    
    
                # Adjust coordinates
                latitudes, longitudes = adjust_coordinates(latitudes, longitudes)
                
                # Normalize coordinates
                latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
                
                inputs = {
                    'text_embeddings': text_embeddings,
                    'color_histograms': color_histograms
                    
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


            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={'loss': val_loss, 'mse': val_mse})

        # Testing
        epoch_test_mses = []
        epoch_test_lat_mses = []
        epoch_test_long_mses = []
        test_loss, test_mse = 0, 0
        
        
        
        with open(os.path.join(save_dir, 'preds_test_vs_actual.csv'), 'w', newline='') as csvfile:
            fieldnames = ['Predicted Latitude', 'Predicted Longitude', 'True Latitude', 'True Longitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for batch in range(min(BATCH_OVERRIDE,len(test_loader.tfrecord_files)-1)):
                print(f"Testing batch {batch + 1}/{len(test_loader.tfrecord_files)}")

                stuff = test_loader.get_next_batch()
                if stuff is None:
                    continue
                latitudes,longitudes,text_embeddings, color_histograms  = stuff
                # Adjust coordinates
                latitudes, longitudes = adjust_coordinates(latitudes, longitudes)
                
                # Normalize coordinates
                latitudes, longitudes = normalize_coordinates(latitudes, longitudes, bounding_box)
                
                inputs = {
                    'text_embeddings': text_embeddings,
                    'color_histograms': color_histograms,
                    
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
                predictions = model.predict(inputs)
                print(epoch, epochs)
                if epoch == epochs-1:
                    for true_lat, true_lon, pred in zip(latitudes, longitudes, predictions):
                        writer.writerow({
                            'Predicted Latitude': pred[0],
                            'Predicted Longitude': pred[1],
                            'True Latitude': true_lat.numpy(),
                            'True Longitude': true_lon.numpy()
                        })

                
                

            test_loss /= len(test_loader.tfrecord_files)
            test_mse /= len(test_loader.tfrecord_files)
            print(f"Test Loss: {test_loss}, Test MSMAIN_BRANCH_SHUFFLEDMAIN_BRANCH_SHUFFLEDE: {test_mse}")

            test_writer.writerow(epoch_test_mses)
            test_lat_writer.writerow(epoch_test_lat_mses)
            test_long_writer.writerow(epoch_test_long_mses)
    model.save(f'asdfsda{run_id}/happymodel222.h5')
                

if __name__ == "__main__":
    train_dir = "./embedded_datasets/SEGS_SINGLE_PER_ROW/train"
    val_dir = "./embedded_datasets/SEGS_SINGLE_PER_ROW/val"
    test_dir = "./embedded_datasets/SEGS_SINGLE_PER_ROW/test" # Adjust these ones for sanity check and development
    batch_size = 512  # Adjust as needed
    epochs = 1  # Adjust as needed
    batch_size = 512  # Adjust as needed
    epochs = 50  # Adjust as needed
    save_dir = "./SEGS_2_ATTN"  # User-specified directory for saving files
    
    train_model(train_dir, val_dir, test_dir, batch_size, epochs, candidate_area_lat, candidate_area_lon, save_dir)