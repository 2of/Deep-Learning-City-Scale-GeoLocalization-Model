import pandas as pd
import sys
from OBJ.YOLO_wrapper import YOLOWrapper
from OCR.easyOCR_wrapper import EasyOCRWrapper
from ColourGram.HISTOGRAM_wrapper import HISTOGRAM_WRAPPER
import os
from PIL import Image, UnidentifiedImageError
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

OUT_TF_RECORDS_DIR = "./data/tfrecords/"
df = pd.read_pickle('./chicago.pkl')
print("LOADED DF ")
print(df)
BATCH_SIZE = 6
YOLO = YOLOWrapper(model_path="./models/billboardsoly100epoch.pt")
EOCR = EasyOCRWrapper()
HIST = HISTOGRAM_WRAPPER()
PATH_TO_DATA = "/home/noahk/Desktop/CompleteDataset"
SAMPLE_IMG_PATH = "./sample_data/samplestreetviews/chicago.png"

def read_pkl_as_dataframe(file_path):
    try:
        # Load the DataFrame from the .pkl file
        dataframe = pd.read_pickle(file_path)
        print("Pickle file successfully read!")
        return dataframe
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_batch(df, start, end):
    return df.iloc[start:end]

def get_image_df(df):
    # Filter out rows where 'is_retrieved' is False
    df_filtered = df[df['is_retrieved'] == True]
    df_filtered = df
    print(f"Filtered DataFrame size: {len(df_filtered)}")
    
    # Lists to store the images and corresponding metadata
    images = []
    ids = []
    latitudes = []
    longitudes = []

    for index, row in df_filtered.iterrows():
        file_path = os.path.join(PATH_TO_DATA, f"{row['id']}.jpg")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as file:
                    img = Image.open(io.BytesIO(file.read()))
                    images.append(img)
                    ids.append(row['id'])
                    latitudes.append(row['latitude'])
                    longitudes.append(row['longitude'])
            except UnidentifiedImageError:
                print(f"Cannot identify image file: {file_path}")
        else:
            print("path does not exist: ", file_path)
    
    # Ensure all lists have the same length
    if len(images) == len(ids) == len(latitudes) == len(longitudes):
        images_df = pd.DataFrame({
            'id': ids,
            'image': images,
            'latitude': latitudes,
            'longitude': longitudes
        })
    else:
        raise ValueError("Mismatch in lengths of images and metadata lists")

    return images_df

def create_tf_example(id, latitude, longitude, text_list, text_embeddings, color_histograms, num_detections):
    flattened_text_list = [t for sublist in text_list for t in sublist]
    feature = {
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
        'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[latitude])),
        'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[longitude])),
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.encode('utf-8') for t in flattened_text_list])),
        'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=text_embeddings.flatten().tolist())),
        'color_histograms': tf.train.Feature(float_list=tf.train.FloatList(value=color_histograms.flatten().tolist())),
        'num_detections': tf.train.Feature(int64_list=tf.train.Int64List(value=[num_detections]))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def save_to_tfrecord(writer, df, text_list, text_embeddings_list, color_histograms_list, num_detections_list):
    for index, row in df.iterrows():
        id = row['id']
        latitude = row['latitude']
        longitude = row['longitude']
        
        # Get the corresponding text, text embeddings, color histograms, and number of detections
        text = text_list[index]
        text_embeddings = text_embeddings_list[index]
        color_histograms = color_histograms_list[index]
        num_detections = num_detections_list[index]
        
        tf_example = create_tf_example(id, latitude, longitude, text, text_embeddings, color_histograms, num_detections)
        writer.write(tf_example.SerializeToString())

def main(batch_size):
    all_texts = []
    all_text_embeddings = []
    all_color_histograms = []
    all_num_detections = []
    all_df = pd.DataFrame()
    no_detections_count = 0

    num_batches = (len(df) + batch_size - 1) // batch_size  # Calculate the number of batches, including the last smaller batch
    print(f"Total number of batches: {num_batches} for df with size {len(df)}")

    for i in range(82,83):
        print(f"Processing batch {i+1}/{num_batches}")
        start_idx = batch_size * i
        end_idx = min(batch_size * (i + 1), len(df))  # Ensure the end index does not exceed the length of the DataFrame
        current_batch = get_batch(df, start_idx, end_idx)
        print(f"Current batch size: {len(current_batch)}")
        images_df = get_image_df(current_batch)
        print(f"Batch {i+1} size after filtering: {len(images_df)}")
        
        for index, row in images_df.iterrows():
            image = row['image']
            localfeatures, labels_tensor, distance_matrix = YOLO.get_objects_and_labels(image)

            if localfeatures is not None:
                text, text_embeddings = EOCR.predict_and_embed_from_group_as_tensor(tensor_of_images=localfeatures)
                segment_colour_embeds = HIST.get_color_histogram_tensor_stack(localfeatures)
                print ("TEXT_EMBEDDING_SHAPE" , text_embeddings.shape)
                print(segment_colour_embeds.shape)
                all_texts.append(text)
                all_text_embeddings.append(text_embeddings)
                all_color_histograms.append(segment_colour_embeds)
                all_num_detections.append(len(localfeatures))
                all_df = pd.concat([all_df, pd.DataFrame([row])], ignore_index=True)
            else:
                no_detections_count += 1

        # Save the current batch to a separate TFRecord file
        batch_filename = os.path.join(OUT_TF_RECORDS_DIR, f"batch_{i+1}.tfrecord")
        
        # sys.exit()
        # with tf.io.TFRecordWriter(batch_filename) as writer:
            # save_to_tfrecord(writer, all_df, all_texts, all_text_embeddings, all_color_histograms, all_num_detections)
        
        # Clear memory for the next batch
        all_texts.clear()
        all_text_embeddings.clear()
        all_color_histograms.clear()
        all_num_detections.clear()
        all_df = pd.DataFrame()

    print(f"Total images with no detections: {no_detections_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and save to TFRecord.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    args = parser.parse_args()
    main(args.batch_size)