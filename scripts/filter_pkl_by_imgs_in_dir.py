import pandas as pd
import os

# Function to remove rows from the DataFrame where the image file does not exist
def remove_missing_images(df, image_dir):
    existing_ids = []
    for index, row in df.iterrows():
        image_path = os.path.join(image_dir, f"{row['id']}.jpg")
        if os.path.exists(image_path):
            existing_ids.append(row['id'])
        else:
            print(f"Image not found: {image_path}")

    # Filter the DataFrame to keep only rows with existing images
    df_filtered = df[df['id'].isin(existing_ids)]
    return df_filtered

# Load the pickle file
pickle_file = "./chicago.pkl"
df = pd.read_pickle(pickle_file)

# Directory containing the images
image_dir = "./data/unseen_images/small"

# Remove rows with missing images
df_filtered = remove_missing_images(df, image_dir)

# Save the filtered DataFrame back to a pickle file
filtered_pickle_file = "./only_in_dir4.pkl"
df_filtered.to_pickle(filtered_pickle_file)


print(df_filtered)
print(f"Filtered DataFrame saved to {filtered_pickle_file}")
