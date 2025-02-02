import pandas as pd
import sys
from OBJ.YOLO_wrapper import YOLOWrapper
from OBJ.YOLO_OBJ_WRAPPER import YOLO_OBJ_Wrapper
from OCR.easyOCR_wrapper import EasyOCRWrapper
from ColourGram.HISTOGRAM_wrapper import HISTOGRAM_WRAPPER
import os
from PIL import Image, UnidentifiedImageError
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse


OUT_TF_RECORDS_DIR = "./data/unseen_images"
df = pd.read_pickle('./data/unseen_images/only_in_dir4.pkl')
PATH_TO_DATA = "./data/unseen_images/small"
SAMPLE_IMG_PATH = "./sample_data/samplestreetviews/chicago.png"






HYDRANTS_CLASSES = {0: 1}
CROSSINGS_CLASSES = {0: 0}



print("LOADED DF ")
print(df)
BATCH_SIZE = 512
YOLO_CROSSINGS = YOLO_OBJ_Wrapper(model_path="./models/CROSSINGS.pt", class_mapping=CROSSINGS_CLASSES)
# YOLO_LIGHTS = YOLO_OBJ_Wrapper(model_path="./models/LAMPS.pt")
# YOLO_TRAFFIC_LIGHTS = YOLO_OBJ_Wrapper(model_path="./models/TLIGHTS.pt") #not needed
YOLO_HYDRANTS = YOLO_OBJ_Wrapper(model_path="./models/HYDRANTS.pt",class_mapping=HYDRANTS_CLASSES)



TLIGHTS_ETC_CLASSES = {9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 74: 7,75:8, 58:9}
YOLO_TLIGHTS_STOPSIGNS_ETC = YOLO_OBJ_Wrapper(model_path="./models/yolo11n.pt", filters=[9, 10, 11, 12, 13, 74,58,75],class_mapping=TLIGHTS_ETC_CLASSES) # this is the standard yolov11 model (oobe :) ) 

EOCR = EasyOCRWrapper()
HIST = HISTOGRAM_WRAPPER()




def read_pkl_as_dataframe(file_path):
    try:
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
    df_filtered = df[df['is_retrieved'] == True]
    df_filtered = df
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



def save_to_tfrecord(writer, df, text_list, text_embeddings_list, color_histograms_list, num_detections_list,
                     stacked_class_names_vectors_list, stacked_bboxes_list, stacked_confidences_list):
    print("Saving to TFRecord")
    for index, row in df.iterrows():
        id = row['id']
        latitude = row['latitude']
        longitude = row['longitude']
        text = text_list[index]
        text_embeddings = text_embeddings_list[index]
        color_histograms = color_histograms_list[index]
        num_detections = num_detections_list[index]
        
        stacked_class_names_vector = stacked_class_names_vectors_list[index]
        stacked_bboxes = stacked_bboxes_list[index]
        stacked_confidences = stacked_confidences_list[index]
        
        tf_example = create_tf_example(id, latitude, longitude, text, text_embeddings, color_histograms,
                                       num_detections, stacked_class_names_vector, stacked_bboxes,
                                       stacked_confidences)
        writer.write(tf_example.SerializeToString())
        
        
        
        
def update_class_names(tensor, class_dict):
    
    class_ids = tensor[:, 0].numpy().tolist()
    counts = tensor[:, 1].numpy().tolist()
    updated_class_ids = [class_dict.get(int(class_id), class_id) for class_id in class_ids]
    return tf.convert_to_tensor(list(zip(updated_class_ids, counts)), dtype=tf.float32)
def replace_values_in_tensor(tensor, value_dict):
    if tf.size(tensor) == 0:
        return tensor

    tensor_np = tensor.numpy()

    for key, value in value_dict.items():
        tensor_np[tensor_np[:, 0] == key, 0] = value

    result_tensor = tf.convert_to_tensor(tensor_np, dtype=tf.float32)
    
    return result_tensor



def insert_values_into_tensor(length=8, *tensors):
    result_tensor = tf.zeros([length], dtype=tf.float32)
    
    for tensor in tensors:
        for row in tensor:
            index = int(row[0])
            value = float(row[1])  # Ensure the value is a float
            if index < length:
                result_tensor = tf.tensor_scatter_nd_update(result_tensor, [[index]], [value])
    
    return result_tensor

def create_tf_example(id, latitude, longitude, text_list, text_embeddings, color_histograms, num_detections,
                      stacked_class_names_vector, stacked_bboxes, stacked_confidences):
    flattened_text_list = [t for sublist in text_list for t in sublist]
    feature = {
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
        'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[latitude])),
        'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[longitude])),
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.encode('utf-8') for t in flattened_text_list])),
        'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=text_embeddings.numpy().flatten().tolist())),
        'color_histograms': tf.train.Feature(float_list=tf.train.FloatList(value=color_histograms.numpy().flatten().tolist())),
        'num_detections': tf.train.Feature(int64_list=tf.train.Int64List(value=[num_detections])),
        'stacked_class_names_vector': tf.train.Feature(int64_list=tf.train.Int64List(value=stacked_class_names_vector.numpy().tolist())),
        'stacked_bboxes': tf.train.Feature(float_list=tf.train.FloatList(value=stacked_bboxes.numpy().flatten().tolist())),
        'stacked_confidences': tf.train.Feature(float_list=tf.train.FloatList(value=stacked_confidences.numpy().tolist()))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))
def main(batch_size):
    num_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Total number of batches: {num_batches} for df with size {len(df)}")

    all_texts = []
    all_text_embeddings = []
    all_color_histograms = []
    all_num_detections = []
    all_stacked_class_names_vectors = []
    all_stacked_bboxes = []
    all_stacked_confidences = []
    all_df = pd.DataFrame()
    no_detections_count = 0

    for i in range(num_batches):
        print(f"Processing batch {i+1}/{num_batches}")
        start_idx = batch_size * i
        end_idx = min(batch_size * (i + 1), len(df))
        current_batch = get_batch(df, start_idx, end_idx)
        images_df = get_image_df(current_batch)
        print(f"Batch {i+1} size after filtering: {len(images_df)}")
        
        for index, row in images_df.iterrows():
            image = row['image']
            text, text_embedding = EOCR.predict_and_embed(image)

            tensor_class_names_vector1, tensor_bboxes1, tensor_confidences1 = YOLO_TLIGHTS_STOPSIGNS_ETC.predict(image)
            tensor_class_names_vector2, tensor_bboxes2, tensor_confidences2 = YOLO_CROSSINGS.predict(image)

            tensor_class_names_vector1 = tensor_class_names_vector1.numpy()
            tensor_bboxes1 = tensor_bboxes1.numpy()
            tensor_confidences1 = tensor_confidences1.numpy()

            tensor_class_names_vector2 = tensor_class_names_vector2.numpy()
            tensor_bboxes2 = tensor_bboxes2.numpy()
            tensor_confidences2 = tensor_confidences2.numpy()

            class_names_vectors = []
            bboxes_vectors = []
            confidences_vectors = []

            if tensor_class_names_vector1.size > 0:
                class_names_vectors.append(tensor_class_names_vector1)
                bboxes_vectors.append(tensor_bboxes1)
                confidences_vectors.append(tensor_confidences1)

            if tensor_class_names_vector2.size > 0:
                class_names_vectors.append(tensor_class_names_vector2)
                bboxes_vectors.append(tensor_bboxes2)
                confidences_vectors.append(tensor_confidences2)

            stacked_class_names_vector = tf.concat(class_names_vectors, axis=0) if class_names_vectors else tf.zeros((0,), dtype=tf.int32)
            stacked_bboxes = tf.concat(bboxes_vectors, axis=0) if bboxes_vectors else tf.zeros((0, 4), dtype=tf.float32)
            stacked_confidences = tf.concat(confidences_vectors, axis=0) if confidences_vectors else tf.zeros((0,), dtype=tf.float32)

            colour_embedding = HIST.get_color_histogram_tensor(image)

            all_texts.append(text)
            all_text_embeddings.append(text_embedding)
            all_color_histograms.append(colour_embedding)
            all_num_detections.append(len(stacked_class_names_vector))
            all_stacked_class_names_vectors.append(stacked_class_names_vector)
            all_stacked_bboxes.append(stacked_bboxes)
            all_stacked_confidences.append(stacked_confidences)
            all_df = pd.concat([all_df, pd.DataFrame([row])], ignore_index=True)

    tfrecord_filename = os.path.join(OUT_TF_RECORDS_DIR, "predictions21_SMALL.tfrecord")
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        save_to_tfrecord(writer, all_df, all_texts, all_text_embeddings, all_color_histograms, all_num_detections,
                        all_stacked_class_names_vectors, all_stacked_bboxes, all_stacked_confidences)

    print(f"Total images with no detections: {no_detections_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and save to TFRecord.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    args = parser.parse_args()
    main(args.batch_size)