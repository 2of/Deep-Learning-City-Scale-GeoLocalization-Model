'''
// Quick utility to downlaod to local dir yolo model

'''


import os
from ultralytics import YOLO

def download_model(save_directory='./models', model_name='yolo11n.pt'):
    os.makedirs(save_directory, exist_ok=True)

    # Load the model (this will automatically download the pre-trained model if not found)
    model = YOLO(model_name)

    # Create the full save path
    save_path = os.path.join(save_directory, model_name)

    # Save the model locally in the specified directory
    model.save(save_path)
    
    print(f"Model downloaded and saved to: {save_path}")

    return save_path  

