import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import seaborn as sns
import sys
# Load data
base_path = 'OVERALL_ATTN_WITH_STATS'
epochs = range(50)
layers = ['batch_normalization_1', 'batch_normalization_2','dense', 'dense_1', 'dense_2', 'dense_3', 'dense_4', 'dense_5', 'multi_head_attention']

data = {layer: {'hist': [], 'bins': []} for layer in layers}

for epoch in epochs:
    for layer in layers:
        print(layer)
        bias_file = os.path.join(base_path, f'epoch_{epoch}', f'{layer}_biases_histogram.json')
        weight_file = os.path.join(base_path, f'epoch_{epoch}', f'{layer}_weights_histogram.json')
        
        with open(bias_file, 'r') as f:
            bias_data = json.load(f)
        with open(weight_file, 'r') as f:
            weight_data = json.load(f)
        
        data[layer]['hist'].append(bias_data['hist'])
        data[layer]['bins'].append(bias_data['bins'])

# Combine data
for layer in layers:
    data[layer]['hist'] = np.array(data[layer]['hist'])
    data[layer]['bins'] = np.array(data[layer]['bins'])

# Visualization and Analysis
for layer in layers:
    hist = data[layer]['hist']
    bins = data[layer]['bins'][0]

    X, Y = np.meshgrid(bins[:-1], np.arange(len(epochs)))
    Z = hist

    # Smooth the data
    Z_smoothed = gaussian_filter(Z, sigma=1)

    # Flip the Y-axis (epochs)
    Y = np.flip(Y, axis=0)
    Z_smoothed = np.flip(Z_smoothed, axis=0)
    # sys.exit()
    # Create 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_smoothed, cmap='viridis', alpha=1, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.contour(X, Y, Z_smoothed, zdir='z', offset=Z_smoothed.min(), cmap='viridis', alpha=0.5)
    ax.set_xlabel('Weight Distribution', fontsize=12, labelpad=10)
    ax.set_ylabel('Training Epochs', fontsize=12, labelpad=10)
    ax.set_zlabel('Frequency', fontsize=12, labelpad=10)
    ax.set_title(f'Evolution of Weight Distribution in {layer.replace("_", " ").title()}', fontsize=14, pad=20)
    ax.set_box_aspect([2, 1, 1])  # Stretch the x-axis
    ax.view_init(elev=25, azim=45)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Frequency', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Plot Mean Weight Over Epochs
    mean_weights = np.mean(Z_smoothed, axis=1)
    plt.plot(epochs, mean_weights)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Weight Value')
    plt.title(f'Mean Weight Value Over Epochs for {layer.replace("_", " ").title()}')
    plt.show()

    # Plot Standard Deviation of Weights Over Epochs
    std_weights = np.std(Z_smoothed, axis=1)
    plt.plot(epochs, std_weights)
    plt.xlabel('Epochs')
    plt.ylabel('Standard Deviation of Weights')
    plt.title(f'Weight Spread Over Epochs for {layer.replace("_", " ").title()}')
    plt.show()

    # Plot Min and Max Weight Values Over Epochs
    min_weights = np.min(Z_smoothed, axis=1)
    max_weights = np.max(Z_smoothed, axis=1)
    plt.plot(epochs, min_weights, label='Min Weight')
    plt.plot(epochs, max_weights, label='Max Weight')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.title(f'Min and Max Weight Values Over Epochs for {layer.replace("_", " ").title()}')
    plt.legend()
    plt.show()

    # Plot Fraction of Near-Zero Weights
    near_zero_threshold = 0.01
    near_zero_fraction = np.mean(np.abs(Z_smoothed) < near_zero_threshold, axis=1)
    plt.plot(epochs, near_zero_fraction)
    plt.xlabel('Epochs')
    plt.ylabel('Fraction of Near-Zero Weights')
    plt.title(f'Effect of Regularization Over Epochs for {layer.replace("_", " ").title()}')
    plt.show()

    # Plot Outlier Weights Over Epochs
    outlier_threshold = 2.0
    outlier_count = np.sum(np.abs(Z_smoothed) > outlier_threshold, axis=1)
    plt.plot(epochs, outlier_count)
    plt.xlabel('Epochs')
    plt.ylabel('Number of Outliers')
    plt.title(f'Outlier Weights Over Epochs for {layer.replace("_", " ").title()}')
    plt.show()

    # Plot Rate of Change of Mean Weights
    rate_of_change = np.diff(mean_weights)
    plt.plot(epochs[1:], rate_of_change)
    plt.xlabel('Epochs')
    plt.ylabel('Rate of Change of Mean Weights')
    plt.title(f'Learning Rate Impact on Weight Updates for {layer.replace("_", " ").title()}')
    plt.show()

    # Compare Initial and Final Weight Distributions
    initial_weights = Z_smoothed[0, :]
    final_weights = Z_smoothed[-1, :]
    plt.hist(initial_weights, bins=30, alpha=0.5, label='Initial Weights')
    plt.hist(final_weights, bins=30, alpha=0.5, label='Final Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title(f'Initial vs Final Weight Distribution for {layer.replace("_", " ").title()}')
    plt.legend()
    plt.show()

# Layer-Wise Mean Weight Comparison
layer_means = {}
for layer in layers:
    layer_means[layer] = np.mean(data[layer]['hist'], axis=1)

for layer, means in layer_means.items():
    plt.plot(epochs, means, label=layer.replace("_", " ").title())

plt.xlabel('Epochs')
plt.ylabel('Mean Weight Value')
plt.title('Layer-Wise Mean Weight Comparison')
plt.legend()
plt.show()

# Layer Correlation Heatmap
layer_correlations = np.corrcoef([np.mean(data[layer]['hist'], axis=1) for layer in layers])
sns.heatmap(layer_correlations, annot=True, xticklabels=[layer.replace("_", " ").title() for layer in layers], yticklabels=[layer.replace("_", " ").title() for layer in layers])
plt.title('Correlation Between Layer Weight Distributions')
plt.show()

# KDE Plots for Weight Distributions
for epoch in [0, 25, 49]:  # Compare initial, middle, and final epochs
    sns.kdeplot(Z_smoothed[epoch, :], label=f'Epoch {epoch}')

plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.title('Weight Distribution Evolution')
plt.legend()
plt.show()