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

df = pd.read_pickle('./data/unseen_images/only_in_dir.pkl')

HYDRANTS_CLASSES = {0: 1}
CROSSINGS_CLASSES = {0: 0}

print("LOADED DF ")
print(df)
BATCH_SIZE = 512
YOLO_CROSSINGS = YOLO_OBJ_Wrapper(model_path="./models/CROSSINGS.pt", class_mapping=CROSSINGS_CLASSES)
YOLO_HYDRANTS = YOLO_OBJ_Wrapper(model_path="./models/HYDRANTS.pt", class_mapping=HYDRANTS_CLASSES)

TLIGHTS_ETC_CLASSES = {9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 74: 7, 75: 8, 58: 9}
YOLO_TLIGHTS_STOPSIGNS_ETC = YOLO_OBJ_Wrapper(model_path="./models/yolo11n.pt", filters=[9, 10, 11, 12, 13, 74, 58, 75], class_mapping=TLIGHTS_ETC_CLASSES)

EOCR = EasyOCRWrapper()
HIST = HISTOGRAM_WRAPPER()
PATH_TO_DATA = "./data/unseen_images"

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
            value = float(row[1])
            if index < length:
                result_tensor = tf.tensor_scatter_nd_update(result_tensor, [[index]], [value])
    
    return result_tensor

def load_data(batch_size):
    num_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Total number of batches: {num_batches} for df with size {len(df)}")

    all_data = []

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

            data_entry = {
                'id': row['id'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'text': text,
                'text_embedding': text_embedding,
                'color_histogram': colour_embedding,
                'num_detections': len(stacked_class_names_vector),
                'stacked_class_names_vector': stacked_class_names_vector,
                'stacked_bboxes': stacked_bboxes,
                'stacked_confidences': stacked_confidences
            }

            all_data.append(data_entry)

    return all_data


def predict(data, model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Prepare the data for prediction
    # Assuming the model expects a specific input format, you may need to preprocess the data accordingly
    # Here, we assume the model expects a list of text embeddings and color histograms concatenated
    inputs = []
    for entry in data:
        input_vector = tf.concat([entry['text_embedding'], entry['color_histogram']], axis=0)
        inputs.append(input_vector)

    inputs = tf.stack(inputs)
    print(f"Prepared input data for prediction with shape: {inputs.shape}")

    # Make predictions
    predictions = model.predict(inputs)
    print("Predictions made.")

    # Process predictions
    for i, entry in enumerate(data):
        entry['predicted_latitude'] = predictions[i][0]
        entry['predicted_longitude'] = predictions[i][1]

    return data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and load into memory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    args = parser.parse_args()
    data = load_data(args.batch_size)
    predict(data,"./path_to_save_model/my_model3.h5")
    print("Data loaded into memory.")