import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directories
dir1 = "GRAPHICS2_CSV/SEGS_2_NO_ATTN_2/CSVS"  # Segments with No Attention
dir2 = "GRAPHICS2_CSV/SEGS_2_ATTN_2/CSVS"    # Segments with Attention

# Output directory for saving comparison plots
output_dir = "GRAPHICS2_CSV/COMPARISON_PLOTS"
os.makedirs(output_dir, exist_ok=True)

# Dictionary for meaningful names
meaningful_names = {
    # Layer-specific files
    "conv2d_mean_weights.csv": "Mean Weights (Conv2D Layer)",
    "conv2d_min_max_weights.csv": "Min and Max Weights (Conv2D Layer)",
    "conv2d_smoothed_histogram.csv": "Smoothed Histogram (Conv2D Layer)",
    "conv2d_std_dev.csv": "Standard Deviation (Conv2D Layer)",
    "conv2d_1_mean_weights.csv": "Mean Weights (Conv2D_1 Layer)",
    "conv2d_1_min_max_weights.csv": "Min and Max Weights (Conv2D_1 Layer)",
    "conv2d_1_smoothed_histogram.csv": "Smoothed Histogram (Conv2D_1 Layer)",
    "conv2d_1_std_dev.csv": "Standard Deviation (Conv2D_1 Layer)",
    "dense_1_mean_weights.csv": "Mean Weights (Dense_1 Layer)",
    "dense_1_min_max_weights.csv": "Min and Max Weights (Dense_1 Layer)",
    "dense_1_smoothed_histogram.csv": "Smoothed Histogram (Dense_1 Layer)",
    "dense_1_std_dev.csv": "Standard Deviation (Dense_1 Layer)",
    "dense_2_mean_weights.csv": "Mean Weights (Dense_2 Layer)",
    "dense_2_min_max_weights.csv": "Min and Max Weights (Dense_2 Layer)",
    "dense_2_smoothed_histogram.csv": "Smoothed Histogram (Dense_2 Layer)",
    "dense_2_std_dev.csv": "Standard Deviation (Dense_2 Layer)",
    "multi_head_attention_mean_weights.csv": "Mean Weights (Multi-Head Attention Layer)",
    "multi_head_attention_min_max_weights.csv": "Min and Max Weights (Multi-Head Attention Layer)",
    "multi_head_attention_smoothed_histogram.csv": "Smoothed Histogram (Multi-Head Attention Layer)",
    "multi_head_attention_std_dev.csv": "Standard Deviation (Multi-Head Attention Layer)",
    
    # Aggregated files
    "layer_correlations.csv": "Correlation Between Layer Weight Distributions",
    "layer_mean_comparison.csv": "Layer-Wise Mean Weight Comparison",
    "layer_mean_weights.csv": "Mean Weights Across Layers",
    "mse_summary.csv": "MSE Summary Across Epochs",
    
    # MSE files
    "test_mses_LAT_rate_of_change.csv": "Rate of Change of Test MSE (LAT)",
    "test_mses_LAT_raw_mse.csv": "Raw Test MSE (LAT)",
    "test_mses_LAT_rolling_avg.csv": "Rolling Average of Test MSE (LAT)",
    "test_mses_LONG_rate_of_change.csv": "Rate of Change of Test MSE (LONG)",
    "test_mses_LONG_raw_mse.csv": "Raw Test MSE (LONG)",
    "test_mses_LONG_rolling_avg.csv": "Rolling Average of Test MSE (LONG)",
    "test_mses_OVERALL_rate_of_change.csv": "Rate of Change of Test MSE (Overall)",
    "test_mses_OVERALL_raw_mse.csv": "Raw Test MSE (Overall)",
    "test_mses_OVERALL_rolling_avg.csv": "Rolling Average of Test MSE (Overall)",
    "train_mses_LAT_rate_of_change.csv": "Rate of Change of Training MSE (LAT)",
    "train_mses_LAT_raw_mse.csv": "Raw Training MSE (LAT)",
    "train_mses_LAT_rolling_avg.csv": "Rolling Average of Training MSE (LAT)",
    "train_mses_LONG_rate_of_change.csv": "Rate of Change of Training MSE (LONG)",
    "train_mses_LONG_raw_mse.csv": "Raw Training MSE (LONG)",
    "train_mses_LONG_rolling_avg.csv": "Rolling Average of Training MSE (LONG)",
    "train_mses_OVERALL_rate_of_change.csv": "Rate of Change of Training MSE (Overall)",
    "train_mses_OVERALL_raw_mse.csv": "Raw Training MSE (Overall)",
    "train_mses_OVERALL_rolling_avg.csv": "Rolling Average of Training MSE (Overall)",
    "val_mses_LAT_rate_of_change.csv": "Rate of Change of Validation MSE (LAT)",
    "val_mses_LAT_raw_mse.csv": "Raw Validation MSE (LAT)",
    "val_mses_LAT_rolling_avg.csv": "Rolling Average of Validation MSE (LAT)",
    "val_mses_LONG_rate_of_change.csv": "Rate of Change of Validation MSE (LONG)",
    "val_mses_LONG_raw_mse.csv": "Raw Validation MSE (LONG)",
    "val_mses_LONG_rolling_avg.csv": "Rolling Average of Validation MSE (LONG)",
    "val_mses_OVERALL_rate_of_change.csv": "Rate of Change of Validation MSE (Overall)",
    "val_mses_OVERALL_raw_mse.csv": "Raw Validation MSE (Overall)",
    "val_mses_OVERALL_rolling_avg.csv": "Rolling Average of Validation MSE (Overall)",
}

# Get the list of CSV files in the first directory
csv_files = [f for f in os.listdir(dir1) if f.endswith('.csv')]

# Function to plot pairs of CSV data
def plot_comparison(csv_file, dir1, dir2, output_dir, meaningful_names):
    # Load data from both directories
    df1 = pd.read_csv(os.path.join(dir1, csv_file))
    df2 = pd.read_csv(os.path.join(dir2, csv_file))
    
    # Get the meaningful name for the plot title
    plot_title = meaningful_names.get(csv_file, csv_file.replace(".csv", ""))
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    
    # Handle single-column and multi-column CSV files
    if df1.shape[1] == 1:  # Single column
        plt.plot(df1, label='Segments with No Attention')
        plt.plot(df2, label='Segments with Attention')
    else:  # Multiple columns
        for col in df1.columns:
            plt.plot(df1[col], label=f'Segments with No Attention ({col})')
            plt.plot(df2[col], label=f'Segments with Attention ({col})')
    
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title(f'Comparison of {plot_title}')
    plt.legend()
    plt.grid()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'{csv_file.replace(".csv", "")}_comparison.png')
    plt.savefig(output_file)
    plt.close()
    print(f'Saved comparison plot to {output_file}')

# Iterate over all CSV files and generate comparison plots
for csv_file in csv_files:
    plot_comparison(csv_file, dir1, dir2, output_dir, meaningful_names)

print("All comparison plots generated successfully!")