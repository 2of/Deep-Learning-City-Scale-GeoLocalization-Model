import os
import pandas as pd

# Define the directory containing the files
input_directory = "OVERALL_NO_ATTN_WITH_STATS"
output_directory = os.path.join(input_directory, "oneper")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List of files to process (including the new test files)
files = [
    "train_mses_LAT.csv",
    "train_mses_LONG.csv",
    "train_mses_OVERALL.csv",
    "val_mses_LAT.csv",
    "val_mses_LONG.csv",
    "val_mses_OVERALL.csv",
    "test_mses_LAT.csv",
    "test_mses_LONG.csv",
    "test_mses_OVERALL.csv"
]

# Process each file
for file in files:
    # Construct the full file path
    input_file_path = os.path.join(input_directory, file)
    
    # Read the CSV file without headers
    df = pd.read_csv(input_file_path, header=None)
    
    # Check if the file is one of the test files (single row)
    if file.startswith("test_"):
        # Transpose the DataFrame so each value becomes a new row
        df = df.T  # Transpose the DataFrame
        # Save the transposed DataFrame to a CSV file without headers
        output_file_path = os.path.join(output_directory, file)
        df.to_csv(output_file_path, index=False, header=False)
    else:
        # For non-test files, calculate the row-wise average
        df['Average'] = df.mean(axis=1)
        # Create a new DataFrame with only the average column
        new_df = df[[df.columns[-1]]]  # Select the last column (the 'Average' column)
        # Save the new DataFrame to a CSV file without headers
        output_file_path = os.path.join(output_directory, file)
        new_df.to_csv(output_file_path, index=False, header=False)

print("Processing complete. Files saved in the 'oneper' directory.")