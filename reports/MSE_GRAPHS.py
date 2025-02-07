import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the base directory
base_dir = "OVERALL_ATTN_WITH_STATS"

# Find all CSV files in the directory
csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]

# Store results for table summary
summary = {}

# Define a mapping for file names to plot titles
file_titles = {
    "test_mses_LAT.csv": "Test MSE for LAT",
    "test_mses_LONG.csv": "Test MSE for LONG",
    "test_mses_OVERALL.csv": "Test MSE for Overall",
    "train_mses_LAT.csv": "Training MSE for LAT",
    "train_mses_LONG.csv": "Training MSE for LONG",
    "train_mses_OVERALL.csv": "Training MSE for Overall",
    "val_mses_LAT.csv": "Validation MSE for LAT",
    "val_mses_LONG.csv": "Validation MSE for LONG",
    "val_mses_OVERALL.csv": "Validation MSE for Overall"
}

# Store data for combined plots
data_categories = {"test": [], "train": [], "val": []}

# Process each file
for csv_file in csv_files:
    file_path = os.path.join(base_dir, csv_file)
    
    # Read CSV file
    df = pd.read_csv(file_path, header=None)
    
    # If it's a test file, assume single row where each column is an epoch
    if "test" in csv_file.lower():
        epoch_means = df.iloc[0]  # Use the single row directly
    else:
        epoch_means = df.mean(axis=1)  # Compute mean per epoch for train and val files
    
    # Compute rolling average (trend smoothing)
    rolling_avg = epoch_means.rolling(window=5, min_periods=1).mean()
    
    # Compute rate of change (gradient)
    rate_of_change = np.gradient(epoch_means)
    
    # Identify best epoch (lowest MSE)
    best_epoch = epoch_means.idxmin()
    
    # Compute total variance
    variance = np.var(epoch_means)
    
    # Store final epoch MSE, mean MSE, variance, and best epoch for summary
    summary[csv_file] = {
        "Final MSE": epoch_means.iloc[-1],
        "Mean MSE": epoch_means.mean(),
        "Best Epoch": best_epoch,
        "Best MSE": epoch_means.min(),
        "Variance": variance
    }
    
    # Determine category (test, train, val) for final plot
    for category in data_categories:
        if category in csv_file.lower():
            data_categories[category].append((csv_file, epoch_means))
    
    # Plot individual file with trend analysis
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_means, label="Raw MSE")
    plt.plot(rolling_avg, label="Rolling Avg (5 epochs)", linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    
    # Use the file_titles dictionary to set the title based on the file name
    title = file_titles.get(csv_file, "MSE Over Epochs")
    plt.title(title)
    
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot rate of change
    plt.figure(figsize=(10, 6))
    plt.plot(rate_of_change, label="Rate of Change", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("ΔMSE")
    plt.title(f"Rate of Change - {title}")
    plt.legend()
    plt.grid()
    plt.show()

# Plot combined Test, Train, and Val MSEs if data exists
plt.figure(figsize=(10, 6))
any_data = False
for category, data in data_categories.items():
    if data:  # Only plot if there's data
        any_data = True
        for csv_file, epoch_means in data:
            plt.plot(epoch_means, label=file_titles.get(csv_file, csv_file.replace(".csv", "")))

if any_data:
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Combined MSE Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("No data available for combined MSE plot.")

# Print summary table with additional insights: Mean MSE, Final MSE, and Variance
summary_df = pd.DataFrame.from_dict(summary, orient="index")
summary_df["Average vs Final MSE Difference"] = summary_df["Mean MSE"] - summary_df["Final MSE"]
print(summary_df)