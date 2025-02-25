import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the base directory
base_dir = "OVERALL_ATTN_WITH_STATS"
output_dir = "MSE_Plots"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Find all CSV files in the directory
csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]

# Files to exclude
exclude_files = {"detailed_logs.csv", "preds_test_vs_actual.csv"}

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
    if csv_file in exclude_files:
        continue

    file_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(file_path, header=None)
    
    if "test" in csv_file.lower():
        epoch_means = df.iloc[0]
    else:
        epoch_means = df.mean(axis=1)
    
    rolling_avg = epoch_means.rolling(window=5, min_periods=1).mean()
    rate_of_change = np.gradient(epoch_means)
    best_epoch = epoch_means.idxmin()
    variance = np.var(epoch_means)
    
    summary[csv_file] = {
        "Final MSE": epoch_means.iloc[-1],
        "Mean MSE": epoch_means.mean(),
        "Best Epoch": best_epoch,
        "Best MSE": epoch_means.min(),
        "Variance": variance
    }
    
    for category in data_categories:
        if category in csv_file.lower():
            data_categories[category].append((csv_file, epoch_means))
    
    title = file_titles.get(csv_file, "MSE Over Epochs")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_means, label="Raw MSE")
    plt.plot(rolling_avg, label="Rolling Avg (5 epochs)", linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{csv_file.replace('.csv', '')}_mse_plot.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(rate_of_change, label="Rate of Change", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("ΔMSE")
    plt.title(f"Rate of Change - {title}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{csv_file.replace('.csv', '')}_rate_of_change.png"))
    plt.close()

# Combined Test, Train, and Val MSEs
plt.figure(figsize=(10, 6))
any_data = False
for category, data in data_categories.items():
    if data:
        any_data = True
        for csv_file, epoch_means in data:
            plt.plot(epoch_means, label=file_titles.get(csv_file, csv_file.replace(".csv", "")))
if any_data:
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Combined MSE Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "combined_mse_plot.png"))
    plt.close()

# Additional Graphs
plt.figure(figsize=(10, 6))
for category, data in data_categories.items():
    if data:
        for csv_file, epoch_means in data:
            sns.kdeplot(epoch_means, label=file_titles.get(csv_file, csv_file.replace(".csv", "")))
plt.xlabel("MSE")
plt.ylabel("Density")
plt.title("Distribution of MSEs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "mse_distribution.png"))
plt.close()

plt.figure(figsize=(10, 6))
for category, data in data_categories.items():
    if data:
        for csv_file, epoch_means in data:
            plt.plot(epoch_means.cumsum(), label=file_titles.get(csv_file, csv_file.replace(".csv", "")))
plt.xlabel("Epoch")
plt.ylabel("Cumulative MSE")
plt.title("Cumulative MSE Over Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "cumulative_mse_plot.png"))
plt.close()

plt.figure(figsize=(10, 6))
mse_data = []
labels = []
for category, data in data_categories.items():
    if data:
        for csv_file, epoch_means in data:
            mse_data.append(epoch_means)
            labels.append(file_titles.get(csv_file, csv_file.replace(".csv", "")))
plt.boxplot(mse_data, labels=labels)
plt.xlabel("Dataset")
plt.ylabel("MSE")
plt.title("Box Plot of MSEs")
plt.grid()
plt.savefig(os.path.join(output_dir, "boxplot_mse.png"))
plt.close()

summary_df = pd.DataFrame.from_dict(summary, orient="index")
summary_df["Average vs Final MSE Difference"] = summary_df["Mean MSE"] - summary_df["Final MSE"]
print(summary_df)
