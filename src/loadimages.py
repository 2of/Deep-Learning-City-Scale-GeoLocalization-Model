import tensorflow as tf
import os
import argparse

def load_and_preprocess_image(file_path, target_size=(640, 640)):
    """
    Load and preprocess a single image:
    - Decode the image
    - Resize to target size
    - Normalize pixel values to [0, 1]
    """
    try:
        # Read the image from the file path
        image = tf.io.read_file(file_path)
        # Decode the image and ensure it's 3 channels (RGB)
        image = tf.image.decode_jpeg(image, channels=3)  # You can also use decode_png if necessary
        # Resize to target size
        image = tf.image.resize(image, target_size)
        # Normalize pixel values to range [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def create_dataset(image_dir, batch_size, target_size=(640, 640)):
    """
    Create a tf.data.Dataset for efficient batch loading.
    - image_dir: Directory containing the images
    - batch_size: Number of images per batch
    """
    # Resolve image paths
    image_paths = [
        os.path.join(image_dir, img) for img in os.listdir(image_dir) 
        if img.lower().endswith(('jpg', 'png'))
    ]

    # Debug: Print paths and ensure they are valid
  #  print("Resolved image paths:", image_paths)
    if not image_paths:
        raise ValueError(f"No valid images found in directory: {image_dir}")

    # Create the dataset from the image paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Apply the image loading and processing function using .map()
    dataset = dataset.map(lambda x: tf.py_function(func=load_and_preprocess_image, inp=[x], Tout=tf.float32),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def load_images(image_dir=None, batch_size=2, target_size=(640, 640)):
    """
    A wrapper function to load and preprocess images, making it easy to use in another file.
    - image_dir: Directory containing the images
    - batch_size: Number of images per batch
    - target_size: Resize images to this size
    """
    # If image_dir is not passed as an argument, use the default or hardcoded value
    if image_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
        parent_dir = os.path.dirname(script_dir)  # Get the parent directory (one level up from 'src')
        image_dir = os.path.join(parent_dir, 'res', 'samplestreetviews')  # Correctly join to get the full path
    
    print("Loading from:", image_dir)  # Print to verify the correct path
    return create_dataset(image_dir, batch_size, target_size)


if __name__ == "__main__":
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../res/samplestreetviews'))
    batch_size = 2
    dataset = load_images(image_dir, batch_size)
    print(dataset)

    # Iterate through the dataset to check if images are correctly loaded
    for batch in dataset:
        print("Batch shape:", batch.shape)
