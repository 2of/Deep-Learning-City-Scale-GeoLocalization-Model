import pandas as pd
from download import download_image
import json
import sys

BATCH = 100

# Path to the .pkl file
file_path = './data/GeoJSON/image_data_chicago.pkl'

# Load the DataFrame
df = pd.read_pickle(file_path)

# Display the first few rows of the DataFrame

with open('./KEYS.json', 'r') as key_file:
    keys = json.load(key_file)
    ACCESS_TOKEN = keys['mapillary']
    SAVE_DIRECTORY = './downloaded_images'


def download_batch(batch_df, batch_num):
    """
    Processes a batch of rows, downloading images based on data in the batch.

    Args:
        batch_df (DataFrame): The batch of rows to process.
        batch_num (int): The batch number being processed.
    """
    print(f"Processing batch {batch_num} with {len(batch_df)} rows.")
    for index, row in batch_df.iterrows():
        image_id = row['id']
        if row['is_retrieved']:
            print(f"Already retrieved row {index}")
        else:
            try:
                download_image(image_id=image_id, access_token=ACCESS_TOKEN, directory=SAVE_DIRECTORY)
                df.at[index, 'is_retrieved'] = True  # Mark as retrieved
                print(f"Image {image_id} downloaded successfully.")
            except Exception as e:
                print(f"Failed to process row {index} in batch {batch_num}: {e}")

# Display the number of rows and batch details
num_rows = len(df)
print(f"Number of rows in the DataFrame: {num_rows}")
print(f"BATCH SIZE: {BATCH}")
print(f"Total batches to process: {num_rows // BATCH + (1 if num_rows % BATCH != 0 else 0)}")

# Iterate through the DataFrame in batches
for i in range(num_rows // BATCH + (1 if num_rows % BATCH != 0 else 0)):  # Include last partial batch if exists
    batch_start = i * BATCH
    batch_end = min((i + 1) * BATCH, num_rows)  # Ensure we don't go out of bounds
    batch_df = df.iloc[batch_start:batch_end]
    download_batch(batch_df, i + 1)
    print("from ", batch_start, " - to - " ,batch_end)
    # Save the DataFrame after each batch
    df.to_pickle(file_path)  # Update the .pkl file after processing the batch
    if i >= 20:
        sys.exit()
print("Processing complete.")