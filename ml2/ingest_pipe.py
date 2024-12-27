import pandas as pd
import sys
from OBJ.YOLO_wrapper import YOLOWrapper
from OCR.easyOCR_wrapper import EasyOCRWrapper
import os
from PIL import Image
import io

df = pd.read_pickle('./data/GeoJSON/chicago.pkl')
print("LOADED DF ")
print(df)
BATCH_SIZE = 25
YOLO = YOLOWrapper(model_path="./models/yolo11n.pt")
EOCR = EasyOCRWrapper()


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
df = read_pkl_as_dataframe('./data/GeoJSON/image_data_chicago.pkl')
print(df)




def get_batch(df,start,end):
   return df.iloc[start:end]

def get_image_df(df):
    # Filter out rows where 'is_retrieved' is False
    df_filtered = df[df['is_retrieved'] == True]
    
    # List to store the images
    images = []
    # print("HELLO WORLD")
    # Iterate over the filtered dataframe
    for index, row in df_filtered.iterrows():
        file_path = os.path.join('./downloaded_images', f"{row['id']}.jpg")
        print(file_path)
        # file_path = "downloaded_images/117176343751725.jpg"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                img = Image.open(io.BytesIO(file.read()))
                images.append(img)
    # Create a DataFrame with the images
    
    print(df_filtered['id'])
    images_df = pd.DataFrame({'id': df_filtered['id'], 'image': images})
    print(images_df)
    return images_df


num_batches = len(df) // BATCH_SIZE
print(num_batches)

for i in range (4,num_batches):
    current_batch = get_batch(df,BATCH_SIZE*i, BATCH_SIZE*(i+1))
    print(current_batch)
    images_df = get_image_df(current_batch)
    for index, row in images_df.iterrows():
        image = row['image']
        # results = YOLO.predict(image)
        # print(f"Results for {row['id']}: {results}")
        results_ocr = EOCR.predict(image)
        print(results_ocr)
    sys.exit()
        