import pandas as pd
import sys
from OBJ.YOLO_wrapper import YOLOWrapper
from OCR.easyOCR_wrapper import EasyOCRWrapper
from ColourGram.HISTOGRAM_wrapper import HISTOGRAM_WRAPPER
import os
from PIL import Image, UnidentifiedImageError
import io
import tensorflow as tf
import argparse


'''
This ingests the dataset to segments.
This DOES NOT ingest the dataset for the overall image profile

-----------------------------------------------------------------------------






'''

OUT_TF_RECORDS_DIR = "./data/tfrecords/fivetwelv_second/"
# df = pd.read_pickle('./chicago.pkl')
df = pd.read_pickle('./filtered_chicago.pkl')
print("LOADED DF ")
print(df)
BATCH_SIZE = 512
YOLO = YOLOWrapper(model_path="./models/billboardsoly100epoch.pt")
EOCR = EasyOCRWrapper()
HIST = HISTOGRAM_WRAPPER()
PATH_TO_DATA = "/home/noahk/Desktop/CompleteDataset"

PATH_TO_DATA = "/home/noahk/Desktop/chicago_second_missed"


def read_pkl_as_dataframe(file_path):
    try:
        dataframe = pd.read_pickle(file_path)
        print("Pickle file successfully read!")
        return dataframe
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_image_df(df):
    df_filtered = df[df['is_retrieved'] == True]
    print(f"Filtered DataFrame size: {len(df_filtered)}")
    
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
        
        text = text_list[index]
        text_embeddings = text_embeddings_list[index]
        color_histograms = color_histograms_list[index]
        num_detections = num_detections_list[index]
        
        tf_example = create_tf_example(id, latitude, longitude, text, text_embeddings, color_histograms, num_detections)
        writer.write(tf_example.SerializeToString())

def main(batch_size):
    if not os.path.exists(OUT_TF_RECORDS_DIR):
        os.makedirs(OUT_TF_RECORDS_DIR)

    all_texts = []
    all_text_embeddings = []
    all_color_histograms = []
    all_num_detections = []
    all_df = pd.DataFrame()
    batch_count = 0

    for index, row in df.iterrows():
        file_path = os.path.join(PATH_TO_DATA, f"{row['id']}.jpg")
        if not os.path.exists(file_path):
            print(f"Path does not exist: {file_path}")
            continue

        try:
            with open(file_path, 'rb') as file:
                img = Image.open(io.BytesIO(file.read()))
                localfeatures, labels_tensor, distance_matrix = YOLO.get_objects_and_labels(img)

                if localfeatures is not None:
                    text, text_embeddings = EOCR.predict_and_embed_from_group_as_tensor(tensor_of_images=localfeatures)
                    segment_colour_embeds = HIST.get_color_histogram_tensor_stack(localfeatures)
                    all_texts.append(text)
                    all_text_embeddings.append(text_embeddings)
                    all_color_histograms.append(segment_colour_embeds)
                    all_num_detections.append(len(localfeatures))
                    all_df = pd.concat([all_df, pd.DataFrame([row])], ignore_index=True)
                print(len(all_df))
                if len(all_df) == batch_size:
                    batch_count += 1
                    batch_filename = os.path.join(OUT_TF_RECORDS_DIR, f"batch_{batch_count+203}.tfrecord")
                    with tf.io.TFRecordWriter(batch_filename) as writer:
                        save_to_tfrecord(writer, all_df, all_texts, all_text_embeddings, all_color_histograms, all_num_detections)
                    print(f"Saved batch {batch_count} with {len(all_df)} records to {batch_filename}")
                    all_texts.clear()
                    all_text_embeddings.clear()
                    all_color_histograms.clear()
                    all_num_detections.clear()
                    all_df = pd.DataFrame()

        except UnidentifiedImageError:
            print(f"Cannot identify image file, cannot get handle? : {file_path}")

    if len(all_df) > 0:
        batch_count += 1
        batch_filename = os.path.join(OUT_TF_RECORDS_DIR, f"batch_{batch_count}.tfrecord")
        with tf.io.TFRecordWriter(batch_filename) as writer:
            save_to_tfrecord(writer, all_df, all_texts, all_text_embeddings, all_color_histograms, all_num_detections)
        print(f"Saved final batch with {len(all_df)} records to {batch_filename}")

    print("Processing ! Completed! .")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and save to TFRecord.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for ingest process")
    args = parser.parse_args()
    main(args.batch_size)