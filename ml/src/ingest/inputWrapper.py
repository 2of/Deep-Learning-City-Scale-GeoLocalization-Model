import os
import pandas as pd
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class inputWrapper():
    def __init__(self, csv_path_to_file, image_path_to_dir, image_extension=""):
        # Use absolute paths
        self.csv = os.path.abspath(csv_path_to_file)
        self.image_dir = os.path.abspath(image_path_to_dir)
        self.df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.image_extension = image_extension

    
    def load_all_auto(self):
        self.load_csv()
        print("Loaded CSV to dataframe")
        self.shuffle_indexes()
        print("Shuffled indexes of dataframe")
        self.set_splits()
        print("Access test and train batches with obj.get_train(), obj.get_test()")

    def shuffle_indexes(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def load_csv(self):
        self.df = pd.read_csv(self.csv)

    def set_splits(self, proportion=0.8):
        train_df, test_df = train_test_split(self.df, test_size=(1 - proportion), random_state=42)
        self.train_df = train_df
        self.test_df = test_df
        print(f"Training set size: {len(self.train_df)}")
        print(f"Test set size: {len(self.test_df)}")

    def get_train(self):
        return self.train_df

    def get_test(self):
        return self.test_df

    def load_batch(self, image_dir,  batch_size, batch_num):

        start = batch_num * batch_size
        end = start + batch_size
        batch_df = self.train_df.iloc[start:end]
        image_tensors = []
        truth_tensors = []
        
        for _, row in batch_df.iterrows():
            filename = row['filename']
            lat = row['lat']
            lon = row['long']
            image_path = os.path.join(image_dir, filename)
            
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                image = tf.image.resize(image, [640, 640])
                image = tf.cast(image, tf.float32) / 255.0
                image_tensors.append(image)
                truth_tensors.append((lat, lon))
            else:
                print("Path to image not found:", image_path)
        
        return tf.stack(image_tensors), truth_tensors


# Example usage:
# test = inputWrapper('/absolute/path/to/index.csv', '/absolute/path/to/images', image_extension=".jpg")
# test.load_all_auto()

# # Load a batch of images (e.g., batch size of 2, batch number 0)
# batch_tensor, labels = test.load_batch('/absolute/path/to/images', batch_size=5, batch_num=2)
# print(labels)