import pandas as pd
import numpy as np
import os  # For directory operations

# Step 1: Read the CSV file into a pandas DataFrame
file_path = 'SEGS_2_NO_ATTN/preds_test_vs_actual.csv'
df = pd.read_csv(file_path)

# Define the threshold for saving groups
THRESHOLD = 5  # You can adjust this value as needed

# Create the /predgroups directory if it doesn't exist
output_dir = 'predgroups'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 2: Group rows where the last two columns are identical
grouped = df.groupby(['True Latitude', 'True Longitude'])

# Step 3: Create a list to store the results
results = []
mse_results = {}

# Step 4: Process each group
groups_printed = 0  # Counter to keep track of how many groups have been printed
for group_key, group_data in grouped:
    # Extract the True Latitude and True Longitude (they are the same for the group)
    true_lat, true_lon = group_key
    
    # Get the size of the group (number of detections)
    num_detections = len(group_data)
    
    # Check if the group size exceeds the threshold and we haven't printed enough groups yet
    if num_detections > THRESHOLD and groups_printed < 2:
        print(f"Group with True Latitude: {true_lat}, True Longitude: {true_lon} has {num_detections} detections:")
        print(group_data)
        print("\n" + "="*80 + "\n")  # Separator for readability
        groups_printed += 1
    
    # Save the group to a CSV file only if it has more than THRESHOLD rows
    if num_detections > THRESHOLD:
        group_filename = f"group_{true_lat}_{true_lon}.csv"
        group_filepath = os.path.join(output_dir, group_filename)
        group_data.to_csv(group_filepath, index=False)
    
    # Calculate the average of Predicted Latitude and Predicted Longitude
    avg_pred_lat = group_data['Predicted Latitude'].mean()
    avg_pred_lon = group_data['Predicted Longitude'].mean()
    
    # Calculate the MSE for this group
    mse_lat = ((group_data['Predicted Latitude'] - true_lat) ** 2).mean()
    mse_lon = ((group_data['Predicted Longitude'] - true_lon) ** 2).mean()
    mse_total = (mse_lat + mse_lon) / 2  # Average MSE for latitude and longitude
    
    # Append the result to the list
    results.append([avg_pred_lat, avg_pred_lon, true_lat, true_lon, num_detections, mse_total])
    
    # Store the MSE for this group in a dictionary for averaging later
    if num_detections not in mse_results:
        mse_results[num_detections] = []
    mse_results[num_detections].append(mse_total)

# Step 5: Create a new DataFrame from the results
results_df = pd.DataFrame(results, columns=[
    'Predicted Latitude', 'Predicted Longitude', 'True Latitude', 'True Longitude', '# of detections', 'MSE'
])

# Step 6: Select a few members from groups with 1, 2, 3, ..., 10 detections
selected_samples = []
for num_detections in range(1, 11):
    # Filter groups with the current number of detections
    groups_with_detections = results_df[results_df['# of detections'] == num_detections]
    
    # Select the first few members (e.g., 3 members) from these groups
    selected_samples.append(groups_with_detections.head(3))

# Combine the selected samples into a single DataFrame
selected_samples_df = pd.concat(selected_samples)

# Step 7: Calculate the average MSE per number of detections
average_mse_per_detections = {}
for num_detections, mse_values in mse_results.items():
    average_mse_per_detections[num_detections] = np.mean(mse_values)

# Convert the average MSE results to a DataFrame for better visualization
average_mse_df = pd.DataFrame(list(average_mse_per_detections.items()), columns=['# of detections', 'Average MSE'])

# Step 8: Save the results to CSV files
selected_samples_df.to_csv('selected_samples.csv', index=False)
average_mse_df.to_csv('average_mse_per_detections_SNA.csv', index=False)

print("Selected samples saved to 'selected_samples.csv'")
print("Average MSE per number of detections saved to 'average_mse_per_detections.csv'")
print(f"Groups with more than {THRESHOLD} rows saved to '{output_dir}/' directory.")