from utils import get_batch
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from models import concatModelForSegments
# Suppress TensorFlow logging



'''
Little bit of overhead here as the segs were initially 

lat,long , {all text embeddigns}, {all sign embeddigns colours hists}


but are now one per each ...... 
 (but the code works so )

'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize dictionaries to store MSE values
test_mses = {}
val_mses = {}
train_mses = {}

def train_model(train_dir, val_dir, test_dir, batch_size, epochs):
    # Load the datasets
    print("Loading datasets...")
    train_dataset = get_batch(train_dir, batch_size, shuffle=True, mode='train')
    val_dataset = get_batch(val_dir, batch_size, shuffle=False, mode='val')
    test_dataset = get_batch(test_dir, batch_size, shuffle=False, mode='test')
    
    # Format the datasets
    train_dataset = train_dataset.map(lambda lat, lon, text, color: (({'text_embeddings': text, 'color_histograms': color}), tf.stack([lat, lon], axis=1)))
    val_dataset = val_dataset.map(lambda lat, lon, text, color: (({'text_embeddings': text, 'color_histograms': color}), tf.stack([lat, lon], axis=1)))
    test_dataset = test_dataset.map(lambda lat, lon, text, color: (({'text_embeddings': text, 'color_histograms': color}), tf.stack([lat, lon], axis=1)))
    
    # Initialize the model
    model = concatModelForSegments()
    optimizer = Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mse'])
    
    # Set up TensorBoard callback
    log_dir = "./logs"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    print("HELLLOOOO")
    for inputs, labels in train_dataset.take(1):
        print("SDFDSFSDFSDF")
        print(f"Inputs: {inputs}")
        print(f"Labels: {labels}")
        print(f"Model output shape: {model(inputs).shape}")
        print(f"Labels shape: {labels.shape}")
        print(type(labels))
        print("___________")
    # Training loop with TensorBoard!
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback]
    )
    
    # Evaluate the model on the test dataset
    test_loss, test_mse = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}")

if __name__ == "__main__":
    train_dir = "./data/tfrecords/fivetwelve/train"
    val_dir = "./data/tfrecords/fivetwelve/val"
    test_dir = "./data/tfrecords/fivetwelve/test"
    batch_size = 512  # Adjust as needed
    epochs = 10  # Adjust as you need (although! On my machine its pretty slow! all data is about 2gb in the segment file.)
    train_model(train_dir, val_dir, test_dir, batch_size, epochs)