import pandas as pd
import matplotlib.pyplot as plt
import os

# List of files
files = [
    "test_mses_LAT.csv",
    "test_mses_LONG.csv",
    "test_mses_OVERALL.csv",
    "train_mses_LAT.csv",
    "train_mses_LONG.csv",
    "train_mses_OVERALL.csv",
    "val_directional_error.csv",
    "val_geodesic_distance.csv",
    "val_mae.csv",
    "val_mean_geodesic_error.csv",
    "val_median_absolute_error.csv",
    "val_mses_LAT.csv",
    "val_mses_LONG.csv",
    "val_mses_OVERALL.csv",
    "val_rmse.csv"
]

# Function to read CSV and calculate mean across rows
def read_and_average(file):
    df = pd.read_csv(file)
    df['mean'] = df.mean(axis=1)
    return df

# Read and process each file
data_frames = {file: read_and_average(file) for file in files}

# Plotting the data
for file, df in data_frames.items():
    plt.figure()
    plt.plot(df.index, df['mean'], label='Mean')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Value')
    plt.title(f'Mean Values Across Batches for {os.path.basename(file)}')
    plt.legend()
    plt.savefig(f'{os.path.basename(file)}.png')

print("Graphs have been generated and saved as PNG files in the current directory.")