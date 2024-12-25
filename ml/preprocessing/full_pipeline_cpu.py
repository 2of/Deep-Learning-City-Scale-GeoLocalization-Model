import pandas as pd

from ml.src.obj_detection.YOLO.yolowrapper import *
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


YOLO = YOLOWrapper()
