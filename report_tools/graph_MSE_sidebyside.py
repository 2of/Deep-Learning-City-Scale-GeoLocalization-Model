import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file1 = "SEGS_2_ATTN/test_mses_OVERALL.csv"
file2 = "SEGS_2_NO_ATTN/test_mses_OVERALL.csv"

# Read the CSV files into pandas DataFrames
df_attn = pd.read_csv(file1, header=None)  # Assuming no header in the CSV
df_no_attn = pd.read_csv(file2, header=None)  # Assuming no header in the CSV

# Extract all values from the rows
mse_with_attention = df_attn.values.flatten()  # Flatten the row into a 1D array
mse_without_attention = df_no_attn.values.flatten()  # Flatten the row into a 1D array

# Create a table for comparison
comparison_df = pd.DataFrame({
    "Index": range(1, len(mse_with_attention) + 1),  # Index for each value
    "MSE_with_Attention": mse_with_attention,
    "MSE_without_Attention": mse_without_attention
})

# Display the table
print("Comparison Table:")
print(comparison_df)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(comparison_df["Index"], comparison_df["MSE_with_Attention"], label="With Attention", marker="o")
plt.plot(comparison_df["Index"], comparison_df["MSE_without_Attention"], label="Without Attention", marker="x")
plt.xlabel("Epoch")
# plt.yscale("log")
plt.ylabel("Test MSE")
plt.title("Test MSE Comparison: With vs Without Attention")
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define paths to the CSV files
file_with_attn = "./SEGS_2_ATTN/train_mses_OVERALL.csv"
file_without_attn = "./SEGS_2_NO_ATTN/train_mses_OVERALL.csv"

# Load the CSV files into pandas DataFrames
df_with_attn = pd.read_csv(file_with_attn, header=None)  # No header in the CSV
df_without_attn = pd.read_csv(file_without_attn, header=None)  # No header in the CSV

# Compute the average MSE per epoch (average of each row)
avg_mse_with_attn = df_with_attn.mean(axis=1)  # Average across columns (batches) for each row (epoch)
avg_mse_without_attn = df_without_attn.mean(axis=1)  # Average across columns (batches) for each row (epoch)

# Create a table for comparison
comparison_df = pd.DataFrame({
    "Epoch": range(1, len(avg_mse_with_attn) + 1),  # Epoch numbers
    "Avg_MSE_with_Attention": avg_mse_with_attn,
    "Avg_MSE_without_Attention": avg_mse_without_attn
})

# Display the table
print("Comparison Table (Average MSE per Epoch):")
print(comparison_df)

# Plot the data with a log scale for the y-axis
plt.figure(figsize=(10, 6))
plt.plot(comparison_df["Epoch"], comparison_df["Avg_MSE_with_Attention"], label="With Attention", marker="o")
plt.plot(comparison_df["Epoch"], comparison_df["Avg_MSE_without_Attention"], label="Without Attention", marker="x")
plt.yscale("log")  # Set y-axis to log scale
plt.xlabel("Epoch")
plt.ylabel("Average Train MSE (Log Scale)")
plt.title("Average Train MSE Comparison: With vs Without Attention (Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid for better readability
plt.show()

# Calculate statistics
def calculate_stats(mse_values, model_name):
    avg_mse = np.mean(mse_values)  # Average MSE across all epochs
    rate_of_change = np.mean(np.diff(mse_values))  # Average rate of change (slope) per epoch
    std_dev = np.std(mse_values)  # Standard deviation of MSE values
    return {
        "Model": model_name,
        "Average MSE": avg_mse,
        "Average Rate of Change (per epoch)": rate_of_change,
        "Standard Deviation of MSE": std_dev
    }

# Compute stats for both models
stats_with_attn = calculate_stats(avg_mse_with_attn, "With Attention")
stats_without_attn = calculate_stats(avg_mse_without_attn, "Without Attention")

# Combine stats into a DataFrame for nice formatting
stats_df = pd.DataFrame([stats_with_attn, stats_without_attn])

# Display the stats in a nicely formatted table
print("\nModel Performance Statistics:")
print(stats_df.to_string(index=False))