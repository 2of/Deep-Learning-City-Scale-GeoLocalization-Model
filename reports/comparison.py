import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
file_path = 'SEGS_2_NO_ATTN/preds_test_vs_actual.csv'
df = pd.read_csv(file_path)

# Step 2: Group rows by True Latitude and True Longitude
grouped = df.groupby(['True Latitude', 'True Longitude'])

# Step 3: Store results
results = []
mse_results = {}
detection_counts = []  # Store the number of detections per group
low_count_groups = []  # Store groups with 5 or fewer detections

# Step 4: Process each group
for group_key, group_data in grouped:
    true_lat, true_lon = group_key
    num_detections = len(group_data)

    avg_pred_lat = group_data['Predicted Latitude'].mean()
    avg_pred_lon = group_data['Predicted Longitude'].mean()

    mse_lat = ((group_data['Predicted Latitude'] - true_lat) ** 2).mean()
    mse_lon = ((group_data['Predicted Longitude'] - true_lon) ** 2).mean()
    mse_total = (mse_lat + mse_lon) / 2

    results.append([avg_pred_lat, avg_pred_lon, true_lat, true_lon, num_detections, mse_total])
    detection_counts.append(num_detections)  # Store detection count for histogram

    if num_detections <= 5:
        low_count_groups.append([true_lat, true_lon, num_detections, mse_total])  # Store for table

    if num_detections not in mse_results:
        mse_results[num_detections] = []
    mse_results[num_detections].append(mse_total)

# Step 5: Convert results to DataFrames
results_df = pd.DataFrame(results, columns=[
    'Predicted Latitude', 'Predicted Longitude', 'True Latitude', 'True Longitude', '# of detections', 'MSE'
])
low_count_df = pd.DataFrame(low_count_groups, columns=['True Latitude', 'True Longitude', '# of detections', 'MSE'])

# Step 6: Compute average MSE per number of detections
average_mse_per_detections = {k: np.mean(v) for k, v in mse_results.items()}

# Convert to DataFrame
average_mse_df = pd.DataFrame(list(average_mse_per_detections.items()), columns=['# of detections', 'Average MSE'])

# Step 7: Plot Histogram (Only for Groups with More Than 5 Detections)
filtered_counts = [count for count in detection_counts if count > 5]

plt.figure(figsize=(8, 5))
plt.hist(filtered_counts, bins=range(6, max(filtered_counts) + 2), alpha=0.7, color='c', edgecolor='black')
plt.xlabel("Number of Detections per Group (More Than 5)")
plt.ylabel("Frequency")
plt.title("Distribution of Groups by Number of Detections (Filtered)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(range(6, max(filtered_counts) + 1))  # Ensure integer tick marks
plt.show()

# Step 8: Save results to CSV
average_mse_df.to_csv('average_mse_per_detections_SNA.csv', index=False)
low_count_df.to_csv('low_count_detections.csv', index=False)

# Step 9: Print table for small groups
print("\nGroups with 5 or fewer detections:")
print(low_count_df.to_string(index=False))

print("\nFiltered histogram plotted & data saved.")