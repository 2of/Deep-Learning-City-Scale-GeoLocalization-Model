import os
import tensorflow as tf
import random
from load_entire_tensor_segs import get_tf_record_gen



import tensorflow.keras.backend as K






'''
Oh, where's the tensorflow?!


Essentially, we *cant* shuffle the dataset (shuffle loads in all files in dir and swaps abotu some indexes, which 
as you can imagine, is dreadfully memory hungry)

Shuffle isjust randomized file order (for the 512 row files)



'''


def haversine_loss(y_true, y_pred):
    ''' 
    This might be overkill, may just use euclidean distance instead
    depends on performance on gpu
    
    '''
    lat1, lon1 = y_true[:, 0], y_true[:, 1]
    lat2, lon2 = y_pred[:, 0], y_pred[:, 1]
    lat1, lon1, lat2, lon2 = map(K.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = K.sin(dlat / 2)**2 + K.cos(lat1) * K.cos(lat2) * K.sin(dlon / 2)**2
    c = 2 * K.atan2(K.sqrt(a), K.sqrt(1 - a))
    r = 6371  # Radius of Earth in kilometers
    return r * c

def get_batch_overall(data_dir, batch_size, shuffle=False, mode=''):
    mode = ""
    
    
    assert batch_size <= 512, "Batch size greater than 512 is not implemented or supported (files are just 512 for now anyway)"

    # Set the random seed for reproducibility
    random.seed(123)
    tf.random.set_seed(123)

    mode_dir = os.path.join(data_dir, mode)
    tfrecord_files = [os.path.join(mode_dir, f) for f in os.listdir(mode_dir) if f.endswith('.tfrecord')]

    # Create a list of tuples (filename, starting_point) based on the batch size
    indexed_files = []
    num_entries = 512 // batch_size
    for tfrecord_file in tfrecord_files:
        for i in range(num_entries):
            indexed_files.append((tfrecord_file, i * batch_size))

    print(len(indexed_files))
    if shuffle:
        random.shuffle(indexed_files)

    def _dataset_generator():
        for tfrecord_file, starting_point in indexed_files:
            latitudes, longitudes, text_embeddings, color_histograms = [], [], [], []
            for i, (latitude, longitude, text_embedding, color_histogram) in enumerate(get_tf_record_gen(tfrecord_file)):
                if i >= starting_point and i < starting_point + batch_size:
                    latitudes.append(latitude)
                    longitudes.append(longitude)
                    text_embeddings.append(text_embedding)
                    color_histograms.append(color_histogram)
            if shuffle:
                combined = list(zip(latitudes, longitudes, text_embeddings, color_histograms))
                random.shuffle(combined)
                latitudes, longitudes, text_embeddings, color_histograms = zip(*combined)
            yield (tf.convert_to_tensor(latitudes, dtype=tf.float32),
                   tf.convert_to_tensor(longitudes, dtype=tf.float32),
                   tf.convert_to_tensor(text_embeddings, dtype=tf.float32),
                   tf.convert_to_tensor(color_histograms, dtype=tf.float32))

    dataset = tf.data.Dataset.from_generator(_dataset_generator, output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 12, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256), dtype=tf.float32)
    ))

    return dataset






def get_batch(data_dir, batch_size, shuffle=True, mode=''):
    mode = ""
    """
    Load and shuffle data from the specified directory.

    Args:
        data_dir (str): The base directory containing the tfrecord files.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        mode (str): The mode of the data to load ('train', 'val', 'test').

    Returns:
        tf.data.Dataset: A TensorFlow dataset object.
    """
    assert batch_size <= 512, "Batch size greater than 512 is not implemented or supported (files are just 512 for now anyway)"

    # Set the random seed for reproducibility
    random.seed(123)
    tf.random.set_seed(123)

    mode_dir = os.path.join(data_dir, mode)
    tfrecord_files = [os.path.join(mode_dir, f) for f in os.listdir(mode_dir) if f.endswith('.tfrecord')]

    # Create a list of tuples (filename, starting_point) based on the batch size
    indexed_files = []
    num_entries = 512 // batch_size
    for tfrecord_file in tfrecord_files:
        for i in range(num_entries):
            indexed_files.append((tfrecord_file, i * batch_size))

    print(len(indexed_files))
    if shuffle:
        random.shuffle(indexed_files)

    def _dataset_generator():
        for tfrecord_file, starting_point in indexed_files:
            latitudes, longitudes, text_embeddings, color_histograms = [], [], [], []
            for i, (latitude, longitude, text_embedding, color_histogram) in enumerate(get_tf_record_gen(tfrecord_file)):
                if i >= starting_point and i < starting_point + batch_size:
                    latitudes.append(latitude)
                    longitudes.append(longitude)
                    text_embeddings.append(text_embedding)
                    color_histograms.append(color_histogram)
            if shuffle:
                combined = list(zip(latitudes, longitudes, text_embeddings, color_histograms))
                random.shuffle(combined)
                latitudes, longitudes, text_embeddings, color_histograms = zip(*combined)
            yield (tf.convert_to_tensor(latitudes, dtype=tf.float32),
                   tf.convert_to_tensor(longitudes, dtype=tf.float32),
                   tf.convert_to_tensor(text_embeddings, dtype=tf.float32),
                   tf.convert_to_tensor(color_histograms, dtype=tf.float32))

    dataset = tf.data.Dataset.from_generator(_dataset_generator, output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 12, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256), dtype=tf.float32)
    ))

    return dataset

def test_get_batch(data_dir, batch_size, num_batches=3, mode='train'):
    """
    Test the get_batch function by loading and printing the first few batches.

    Args:
        data_dir (str): The base directory containing the tfrecord files.
        batch_size (int): The number of samples per batch.
        num_batches (int): The number of batches to load and print.
        mode (str): The mode of the data to load ('train', 'val', 'test').
    """
    dataset = get_batch(data_dir, batch_size, shuffle=True, mode=mode)
    for batch in dataset.take(num_batches):
        latitudes, longitudes, text_embeddings, color_histograms = batch
        print(f"Latitudes: {latitudes.shape}")
        print(f"Longitudes: {longitudes.shape}")
        print(f"Text Embeddings Shape: {text_embeddings.shape}")
        print(f"Color Histograms Shape: {color_histograms.shape}")

if __name__ == "__main__":
    data_dir = "./data/tfrecords/fivetwelve"
    batch_size = 512
    test_get_batch(data_dir, batch_size, num_batches=3, mode='train')