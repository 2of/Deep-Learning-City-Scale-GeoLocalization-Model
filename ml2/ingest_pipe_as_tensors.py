import pandas as pd
import sys
from OBJ.YOLO_wrapper import YOLOWrapper
from OCR.easyOCR_wrapper import EasyOCRWrapper
import os
from PIL import Image
import io
import torch
import numpy as np

df = pd.read_pickle('./chicago.pkl')
print("LOADED DF ")
print(df)
BATCH_SIZE = 25
YOLO = YOLOWrapper(model_path="./models/yolo11n.pt")
EOCR = EasyOCRWrapper()
PATH_TO_DATA = "/home/noahk/Desktop/CompleteDataset"


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

# Example usage
print(df)

def get_batch(df, start, end):
    return df.iloc[start:end]
def image_to_tensor( image):
    """
    Convert a PIL image to a tensor.
    
    Args:
        image (PIL.Image): The image to convert.
    
    Returns:
        torch.Tensor: The image as a tensor.
    """
    image = image.convert('RGB')
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)  # Change to (C, H, W) format
    image = image.unsqueeze(0)  # Add batch dimension
    return image
def get_image_df(df):
    # Filter out rows where 'is_retrieved' is False
    df_filtered = df[df['is_retrieved'] == True]
    
    # List to store the images
    images = []
    # Iterate over the filtered dataframe
    for index, row in df_filtered.iterrows():
        file_path = os.path.join(PATH_TO_DATA, f"{row['id']}.jpg")
        print(file_path)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                img = Image.open(io.BytesIO(file.read()))
                images.append(img)
        else:
            print("path does not exist: ", file_path)
    # Create a DataFrame with the images
    images_df = pd.DataFrame({'id': df_filtered['id'], 'image': images})
    print(images_df)
    return images_df

num_batches = len(df) // BATCH_SIZE
print(num_batches)

for i in range(4, num_batches):
    current_batch = get_batch(df, BATCH_SIZE * i, BATCH_SIZE * (i + 1))
    print(current_batch)
    images_df = get_image_df(current_batch)
    for index, row in images_df.iterrows():
        image = row['image']
        # Convert image to tensor
        image_tensor = image_to_tensor(image)
        print(image_tensor.shape)
        results = YOLO.predict(image_tensor)
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        # results_ocr = EOCR.predict(image_pil)
        # print(results_ocr)
    sys.exit()