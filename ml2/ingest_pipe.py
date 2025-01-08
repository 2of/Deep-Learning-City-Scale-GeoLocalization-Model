import pandas as pd
import sys
from OBJ.YOLO_wrapper import YOLOWrapper
from OCR.easyOCR_wrapper import EasyOCRWrapper
from ColourGram.HISTOGRAM_wrapper import HISTOGRAM_WRAPPER
import os
from PIL import Image
import io
import tensorflow as tf
import matplotlib.pyplot as plt
df = pd.read_pickle('./chicago.pkl')
print("LOADED DF ")
print(df)
BATCH_SIZE = 5
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

# Example usage

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
        file_path = os.path.join(PATH_TO_DATA, f"{row['id']}.jpg")
        print(file_path)
        # file_path = "downloaded_images/117176343751725.jpg"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                img = Image.open(io.BytesIO(file.read()))
                images.append(img)
        else:
            print("path does not exist: ",file_path)
    # Create a DataFrame with the images
    
    print(df_filtered['id'])
    images_df = pd.DataFrame({
        'id': df_filtered['id'],
        'image': images,
        'latitude': df_filtered['latitude'],
        'longitude': df_filtered['longitude']
    })
    print(images_df)
    return images_df

def load_sample_image():
    image = Image.open(SAMPLE_IMG_PATH).convert('RGB')
    localfeatures, labels_tensor, distance_matrix = YOLO.get_objects_and_labels(image)
    show_image(localfeatures)
    
num_batches = len(df) // BATCH_SIZE
print(num_batches)



def compute_and_concat():
    pass
def show_image(img_tensor):
    """
    Display all images in the tensor using plt.show().

    Args:
        img_tensor (torch.Tensor): Tensor containing images with shape [num_detections, 3, height, width].
    """
    if  img_tensor == None:
        return
    num_images = img_tensor.shape[0]

    for i in range(num_images):
        img = img_tensor[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and numpy array
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')
        plt.show()

def show_single_image(image): 
    plt.imshow(image)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# To TFR record:

def create_tf_example(id, latitude, longitude, text_embeddings, color_histograms):
    feature = {
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
        'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[latitude])),
        'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[longitude])),
        'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=text_embeddings.flatten().tolist())),
        'color_histograms': tf.train.Feature(float_list=tf.train.FloatList(value=color_histograms.flatten().tolist()))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


if True:
    for i in range (11,12):
        print("pred", i)
        current_batch = get_batch(df,BATCH_SIZE*i, BATCH_SIZE*(i+1))
        print("Current batch", current_batch)
        images_df = get_image_df(current_batch)
        for index, row in images_df.iterrows():
            print("ITEM:", row['id'])
            
            
            
            
            
            
            
            
            
            
            
            ###extract segments
            image = row['image']
            localfeatures, labels_tensor, distance_matrix = YOLO.get_objects_and_labels(image)

            if localfeatures is not None:
                num_images = localfeatures.shape[0]

                text, text_embeddings = EOCR.predict_and_embed_from_group_as_tensor(tensor_of_images=localfeatures)
                print(text, text_embeddings.shape)
                segment_colour_embeds = HIST.get_color_histogram_tensor_stack(localfeatures)
                print("Row keys:", row.keys())
                create_tf_example(row['id'], row['latitude'], row['longitude'], text_embeddings, segment_colour_embeds)
                print(create_tf_example)
                # for i in range(num_images):
                #     img = localfeatures[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and numpy array
                #     detected_texts, text_embedding_tensor = EOCR.predict_and_embed(img)
                #     print(i, detected_texts)
                #     plt.imshow(img)
                #     plt.title(f"Image {i+1}")
                #     plt.axis('off')
                #     plt.show()

            # None, lets skip the embedding ..... 
            # if localfeatures != None:
            #     print("\n---\nSHAPE OF Localfeatures:" , localfeatures.shape)
                
                
                
            #     ## get the stacked tensors from the easyOCR model: 
                
            #     detected_texts, text_embedding_tensor = EOCR.predict_from_tensor(localfeatures)
                        
            # print(labels_tensor)
            # print("LEN LEN " , len(labels_tensor))
            # show_image(localfeatures)
            # embedded_hists = HIST.create_histogram_embedding(localfeatures)
            # print(embedded_hists.shape)
        
            
            
            
            
            
            
            
            # # # print(f"Results for {row['id']}: {results}")
            # text,text_embedding_tensor = EOCR.predict_and_embed(image)
            # print(text_embedding_tensor.shape)
            # print("i.e.", text)
            # results_hist = HIST.get_color_histogram_tensor(image)
            # print(results_hist.shape)
        sys.exit()
            
            
# if True:
#     load_sample_image()