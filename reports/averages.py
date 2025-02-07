import pandas as pd

# Define the threshold for the minimum number of elements in a group
THRESHOLD = 1  # Change as needed

# Load the first CSV file (SEGS_2_NO_ATTN)
file_path = 'SEGS_2_NO_ATTN/preds_test_vs_actual.csv'
df = pd.read_csv(file_path)

# Group by 'True Latitude' and 'True Longitude'
grouped = df.groupby(['True Latitude', 'True Longitude'])

# Compute averages for each group
averaged_data = grouped.agg({
    'Predicted Latitude': ['mean', 'count'],
    'Predicted Longitude': 'mean'
}).reset_index()

# Flatten column names
averaged_data.columns = ['True Latitude', 'True Longitude', 'Averaged Pred Latitude (SEGS)', 'Count', 'Averaged Pred Longitude (SEGS)']

# Load the second CSV file (OVERALL_NO_ATTN_WITH_STATS)
second_file_path = 'OVERALL_NO_ATTN_WITH_STATS/preds_test_vs_actual.csv'
df_second = pd.read_csv(second_file_path)

# Prepare a list for output data
output_rows = []

# Iterate through the averaged data
for index, row in averaged_data.iterrows():
    true_lat = row['True Latitude']
    true_lon = row['True Longitude']
    avg_pred_lat = row['Averaged Pred Latitude (SEGS)']
    avg_pred_lon = row['Averaged Pred Longitude (SEGS)']
    count = row['Count']

    # Only proceed if the group has more than the threshold count
    if count > THRESHOLD:
        # Find matching rows in the second dataset
        matches = df_second[(df_second['True Latitude'] == true_lat) & (df_second['True Longitude'] == true_lon)]

        # If matches exist, compute the new averages and store them
        if not matches.empty:
            for match_index, match_row in matches.iterrows():
                overall_pred_lat = match_row['Predicted Latitude']
                overall_pred_lon = match_row['Predicted Longitude']
                
                # Compute final average
                final_avg_lat = (avg_pred_lat + overall_pred_lat) / 2
                final_avg_lon = (avg_pred_lon + overall_pred_lon) / 2
                
                # Store the results
                output_rows.append([
                    true_lat, true_lon,
                    avg_pred_lat, avg_pred_lon,
                    overall_pred_lat, overall_pred_lon,
                    final_avg_lat, final_avg_lon
                ])

# Convert to DataFrame
output_df = pd.DataFrame(output_rows, columns=[
    'True Latitude', 'True Longitude',
    'Averaged Pred Latitude (SEGS)', 'Averaged Pred Longitude (SEGS)',
    'Pred Latitude (Overall)', 'Pred Longitude (Overall)',
    'Final Averaged Latitude', 'Final Averaged Longitude'
])

# Save to CSV
output_csv_path = 'averaged_predictions.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"CSV saved: {output_csv_path}")