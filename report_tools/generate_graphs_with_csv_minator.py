import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import seaborn as sns
import pandas as pd

class WeightDistributionAnalyzer:
    def __init__(self, base_path, output_dir, layers, epochs=50):
        self.base_path = base_path
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'CSVS')
        self.epochs = range(epochs)
        self.layers = layers
        self.data = {layer: {'hist': [], 'bins': []} for layer in self.layers}
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        self.load_data()

    def load_data(self):
        for epoch in self.epochs:
            for layer in self.layers:
                bias_file = os.path.join(self.base_path, f'epoch_{epoch}', f'{layer}_biases_histogram.json')
                weight_file = os.path.join(self.base_path, f'epoch_{epoch}', f'{layer}_weights_histogram.json')
                
                with open(bias_file, 'r') as f:
                    bias_data = json.load(f)
                with open(weight_file, 'r') as f:
                    weight_data = json.load(f)
                
                self.data[layer]['hist'].append(bias_data['hist'])
                self.data[layer]['bins'].append(bias_data['bins'])
        
        for layer in self.layers:
            self.data[layer]['hist'] = np.array(self.data[layer]['hist'])
            self.data[layer]['bins'] = np.array(self.data[layer]['bins'])

    def save_plot(self, fig, filename):
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)

    def save_to_csv(self, data, filename):
        filepath = os.path.join(self.csv_dir, filename)
        pd.DataFrame(data).to_csv(filepath, index=False)

    def make_all_graphs(self):
        for layer in self.layers:
            hist = self.data[layer]['hist']
            bins = self.data[layer]['bins'][0]
            X, Y = np.meshgrid(bins[:-1], np.arange(len(self.epochs)))
            Z = hist
            Z_smoothed = gaussian_filter(Z, sigma=1)
            Y = np.flip(Y, axis=0)
            Z_smoothed = np.flip(Z_smoothed, axis=0)
            
            # Save smoothed histogram data to CSV
            self.save_to_csv(Z_smoothed, f'{layer}_smoothed_histogram.csv')
            
            # 3D Plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_smoothed, cmap='viridis', alpha=1, rstride=1, cstride=1, linewidth=0, antialiased=True)
            ax.contour(X, Y, Z_smoothed, zdir='z', offset=Z_smoothed.min(), cmap='viridis', alpha=0.5)
            ax.set_xlabel('Weight Distribution')
            ax.set_ylabel('Training Epochs')
            ax.set_zlabel('Frequency')
            ax.set_title(f'Evolution of {layer}')
            self.save_plot(fig, f'{layer}_3D_plot.png')
            
            # Mean Weight Over Epochs
            mean_weights = np.mean(Z_smoothed, axis=1)
            self.save_to_csv(mean_weights, f'{layer}_mean_weights.csv')
            fig, ax = plt.subplots()
            ax.plot(self.epochs, mean_weights)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Mean Weight')
            ax.set_title(f'Mean Weight Over Epochs for {layer}')
            self.save_plot(fig, f'{layer}_mean_weight.png')
            
            # Standard Deviation Over Epochs
            std_weights = np.std(Z_smoothed, axis=1)
            self.save_to_csv(std_weights, f'{layer}_std_dev.csv')
            fig, ax = plt.subplots()
            ax.plot(self.epochs, std_weights)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Std Dev of Weights')
            ax.set_title(f'Weight Spread for {layer}')
            self.save_plot(fig, f'{layer}_std_dev.png')
            
            # Min and Max Weights
            min_weights = np.min(Z_smoothed, axis=1)
            max_weights = np.max(Z_smoothed, axis=1)
            self.save_to_csv(np.column_stack((min_weights, max_weights)), f'{layer}_min_max_weights.csv')
            fig, ax = plt.subplots()
            ax.plot(self.epochs, min_weights, label='Min')
            ax.plot(self.epochs, max_weights, label='Max')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Weight Value')
            ax.set_title(f'Min and Max Weights for {layer}')
            ax.legend()
            self.save_plot(fig, f'{layer}_min_max_weights.png')
            
            # KDE Plot
            fig, ax = plt.subplots()
            for epoch in [0, len(self.epochs)//2, len(self.epochs)-1]:
                sns.kdeplot(Z_smoothed[epoch, :], label=f'Epoch {epoch}', ax=ax)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Weight KDE Evolution for {layer}')
            ax.legend()
            self.save_plot(fig, f'{layer}_kde_plot.png')
        
        # Layer-Wise Mean Weight Comparison
        fig, ax = plt.subplots()
        layer_mean_data = []
        for layer in self.layers:
            mean_weights = np.mean(self.data[layer]['hist'], axis=1)
            layer_mean_data.append(mean_weights)
            ax.plot(self.epochs, mean_weights, label=layer)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Weight')
        ax.set_title('Layer-Wise Mean Weight Comparison')
        ax.legend()
        self.save_plot(fig, 'layer_mean_comparison.png')
        
        # Save layer-wise mean weights to CSV
        self.save_to_csv(np.column_stack(layer_mean_data), 'layer_mean_comparison.csv')

        # Print statistics to console
        self.print_statistics()

    def print_statistics(self):
        print("\nWeight Distribution Statistics:")
        for layer in self.layers:
            hist = self.data[layer]['hist']
            mean_weights = np.mean(hist, axis=1)
            std_weights = np.std(hist, axis=1)
            min_weights = np.min(hist, axis=1)
            max_weights = np.max(hist, axis=1)
            
            print(f"\nLayer: {layer}")
            print(f"Mean Weight Over Epochs: {mean_weights}")
            print(f"Standard Deviation Over Epochs: {std_weights}")
            print(f"Min Weights Over Epochs: {min_weights}")
            print(f"Max Weights Over Epochs: {max_weights}")

class CorrelatorMinator:
    def __init__(self, source_dir, output_dir, layers, epochs=50):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'CSVS')
        self.layers = layers
        self.epochs = range(epochs)
        self.data = {layer: {'hist': [], 'bins': []} for layer in layers}
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def load_data(self):
        for epoch in self.epochs:
            for layer in self.layers:
                bias_file = os.path.join(self.source_dir, f'epoch_{epoch}', f'{layer}_biases_histogram.json')
                weight_file = os.path.join(self.source_dir, f'epoch_{epoch}', f'{layer}_weights_histogram.json')
                
                with open(bias_file, 'r') as f:
                    bias_data = json.load(f)
                with open(weight_file, 'r') as f:
                    weight_data = json.load(f)
                
                self.data[layer]['hist'].append(bias_data['hist'])
                self.data[layer]['bins'].append(bias_data['bins'])

        # Convert lists to numpy arrays
        for layer in self.layers:
            self.data[layer]['hist'] = np.array(self.data[layer]['hist'])
            self.data[layer]['bins'] = np.array(self.data[layer]['bins'])

    def save_to_csv(self, data, filename):
        filepath = os.path.join(self.csv_dir, filename)
        pd.DataFrame(data).to_csv(filepath, index=False)

    def compute_and_save_correlation_heatmap(self):
        # Compute bin centers (assuming bins are the same for all epochs and layers)
        bin_centers = (self.data[self.layers[0]]['bins'][0][:-1] + self.data[self.layers[0]]['bins'][0][1:]) / 2
        
        # Compute mean weight values for each layer across epochs
        layer_mean_data = []
        for layer in self.layers:
            hist = self.data[layer]['hist']
            mean_weights = np.sum(hist * bin_centers, axis=1) / np.sum(hist, axis=1)
            layer_mean_data.append(mean_weights)
        
        # Save layer mean weights to CSV
        self.save_to_csv(np.column_stack(layer_mean_data), 'layer_mean_weights.csv')
        
        # Compute correlation matrix
        layer_correlations = np.corrcoef(layer_mean_data)
        
        # Save correlation matrix to CSV
        self.save_to_csv(layer_correlations, 'layer_correlations.csv')
        
        # Plot the correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(layer_correlations, annot=True, xticklabels=[layer.replace("_", " ").title() for layer in self.layers], 
                    yticklabels=[layer.replace("_", " ").title() for layer in self.layers])
        plt.title('Correlation Between Layer Weight Distributions')
        
        # Save the plot
        heatmap_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved heatmap to {heatmap_path}")

        # Print correlation matrix to console
        print("\nCorrelation Matrix:")
        print(layer_correlations)

class MSEPlotAnalyzer:
    def __init__(self, base_dir, output_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'CSVS')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        self.exclude_files = {"detailed_logs.csv", "preds_test_vs_actual.csv"}
        self.file_titles = {
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
        self.data_categories = {"test": [], "train": [], "val": []}
        self.summary = {}

    def save_to_csv(self, data, filename):
        filepath = os.path.join(self.csv_dir, filename)
        pd.DataFrame(data).to_csv(filepath, index=False)

    def process_files(self):
        csv_files = [f for f in os.listdir(self.base_dir) if f.endswith(".csv") and f not in self.exclude_files]

        for csv_file in csv_files:
            file_path = os.path.join(self.base_dir, csv_file)
            df = pd.read_csv(file_path, header=None)

            if "test" in csv_file.lower():
                epoch_means = df.iloc[0]
            else:
                epoch_means = df.mean(axis=1)

            rolling_avg = epoch_means.rolling(window=5, min_periods=1).mean()
            rate_of_change = np.gradient(epoch_means)

            best_epoch = epoch_means.idxmin()
            variance = np.var(epoch_means)

            self.summary[csv_file] = {
                "Final MSE": epoch_means.iloc[-1],
                "Mean MSE": epoch_means.mean(),
                "Best Epoch": best_epoch,
                "Best MSE": epoch_means.min(),
                "Variance": variance
            }

            for category in self.data_categories:
                if category in csv_file.lower():
                    self.data_categories[category].append((csv_file, epoch_means))

            title = self.file_titles.get(csv_file, "MSE Over Epochs")
            self._save_plot(epoch_means, rolling_avg, title, csv_file)
            self._save_rate_of_change_plot(rate_of_change, title, csv_file)

            # Save raw MSE data to CSV
            self.save_to_csv(epoch_means, f'{csv_file.replace(".csv", "")}_raw_mse.csv')
            # Save rolling average data to CSV
            self.save_to_csv(rolling_avg, f'{csv_file.replace(".csv", "")}_rolling_avg.csv')
            # Save rate of change data to CSV
            self.save_to_csv(rate_of_change, f'{csv_file.replace(".csv", "")}_rate_of_change.csv')

    def _save_plot(self, epoch_means, rolling_avg, title, csv_file):
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_means, label="Raw MSE")
        plt.plot(rolling_avg, label="Rolling Avg (5 epochs)", linestyle='dashed')
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title(title)
        plt.legend()
        plt.grid()

        if "OVERALL" in csv_file:
            y_min = min(epoch_means.min(), rolling_avg.min())
            y_max = max(epoch_means.max(), rolling_avg.max())
            plt.ylim(y_min - 0.1 * y_min, y_max + 0.1 * y_max)
            plt.ticklabel_format(axis='y', style='plain')

        plt.savefig(os.path.join(self.output_dir, f"{csv_file.replace('.csv', '')}_mse_plot.png"))
        plt.close()

    def _save_rate_of_change_plot(self, rate_of_change, title, csv_file):
        plt.figure(figsize=(10, 6))
        plt.plot(rate_of_change, label="Rate of Change", color='red')
        plt.xlabel("Epoch")
        plt.ylabel("ΔMSE")
        plt.title(f"Rate of Change - {title}")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_dir, f"{csv_file.replace('.csv', '')}_rate_of_change.png"))
        plt.close()

    def generate_combined_plots(self):
        plt.figure(figsize=(10, 6))
        for category, data in self.data_categories.items():
            for csv_file, epoch_means in data:
                plt.plot(epoch_means, label=self.file_titles.get(csv_file, csv_file.replace(".csv", "")))
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Combined MSE Over Epochs")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_dir, "combined_mse_plot.png"))
        plt.close()

    def generate_distribution_plots(self):
        plt.figure(figsize=(10, 6))
        for category, data in self.data_categories.items():
            for csv_file, epoch_means in data:
                sns.kdeplot(epoch_means, label=self.file_titles.get(csv_file, csv_file.replace(".csv", "")))
        plt.xlabel("MSE")
        plt.ylabel("Density")
        plt.title("Distribution of MSEs")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.output_dir, "mse_distribution.png"))
        plt.close()

    def generate_summary_table(self):
        summary_df = pd.DataFrame.from_dict(self.summary, orient="index")
        summary_df["Average vs Final MSE Difference"] = summary_df["Mean MSE"] - summary_df["Final MSE"]
        print("\nMSE Summary Table:")
        print(summary_df)
        # Save summary table to CSV
        summary_df.to_csv(os.path.join(self.csv_dir, 'mse_summary.csv'))
        return summary_df

    def run_analysis(self):
        self.process_files()
        self.generate_combined_plots()
        self.generate_distribution_plots()
        return self.generate_summary_table()

# Usage Example
# base_dir = "path_to_your_csv_files"  # Replace with your directory containing CSV files
# output_dir = "output_plots"  # Replace with your desired output directory

# analyzer = MSEPlotAnalyzer(base_dir, output_dir)
# summary_table = analyzer.run_analysis()

# Usage
# analyzer = MSEPlotAnalyzer("OVERALL_ATTN_WITH_STATS")
# summary_df = analyzer.run_analysis()


# Example usage:
INDIR = 'OVERALL_ATTN_WITH_STATS'
OUTDIR = "GRAPHICS/OVERALL_ATTN"
layers = ['batch_normalization_1', 'batch_normalization_2', 'dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4', 'multi_head_attention']
analyzer_minator = WeightDistributionAnalyzer(INDIR, OUTDIR, layers)
analyzer_minator.make_all_graphs()

correlator_minator = CorrelatorMinator(INDIR, OUTDIR, layers)
correlator_minator.load_data()
correlator_minator.compute_and_save_correlation_heatmap()

Error_Minator = MSEPlotAnalyzer(INDIR, OUTDIR)
Error_Minator.run_analysis()

INDIR = 'OVERALL_NO_ATTN_WITH_STATS'
OUTDIR = "GRAPHICS/OVERALL_NO_ATTN"
layers = ['batch_normalization_1', 'batch_normalization_2', 'dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4']
analyzer_minator = WeightDistributionAnalyzer(INDIR, OUTDIR, layers)
analyzer_minator.make_all_graphs()

correlator_minator = CorrelatorMinator(INDIR, OUTDIR, layers)
correlator_minator.load_data()
correlator_minator.compute_and_save_correlation_heatmap()

Error_Minator = MSEPlotAnalyzer(INDIR, OUTDIR)
Error_Minator.run_analysis()


# # Example usage:
# INDIR = 'SEGS_2_ATTN'
# OUTDIR = "GRAPHICS2_C/OVERALL"
# layers = [ 'dense_1', 'dense_2','conv2d', 'conv2d_1', 'multi_head_attention']
# analyzer_minator = WeightDistributionAnalyzer(INDIR, OUTDIR, layers)
# analyzer_minator.make_all_graphs()

# correlator_minator = CorrelatorMinator(INDIR, OUTDIR, layers)
# correlator_minator.load_data()
# correlator_minator.compute_and_save_correlation_heatmap()

# Error_Minator = MSEPlotAnalyzer(INDIR, OUTDIR)
# Error_Minator.run_analysis()

# INDIR = 'SEGS_2_NO_ATTN'
# OUTDIR = "GRAPHICS2_CSV/SEGS_2_NO_ATTN_2"
# layers = [ 'dense_1', 'dense_2','conv2d', 'conv2d_1']
# analyzer_minator = WeightDistributionAnalyzer(INDIR, OUTDIR, layers)
# analyzer_minator.make_all_graphs()

# correlator_minator = CorrelatorMinator(INDIR, OUTDIR, layers)
# correlator_minator.load_data()
# correlator_minator.compute_and_save_correlation_heatmap()

# Error_Minator = MSEPlotAnalyzer(INDIR, OUTDIR)
# Error_Minator.run_analysis()