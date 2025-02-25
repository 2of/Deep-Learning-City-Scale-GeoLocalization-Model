import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('OVERALL_NO_ATTN_WITH_STATS/my_model.keras')  # Replace with your model file path

# Extract weights and biases
weights = {}
biases = {}
for layer in model.layers:
    if layer.weights:  # Check if the layer has weights
        weights[layer.name] = layer.get_weights()[0]  # Weights
        biases[layer.name] = layer.get_weights()[1]   # Biases

# Function to compute histogram data
def compute_histogram(data, bins=30):
    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges

# Analyze each layer
for layer_name in weights.keys():
    # Flatten weights and biases
    layer_weights = weights[layer_name].flatten()
    layer_biases = biases[layer_name].flatten()

    # Compute histograms
    weight_hist, weight_bins = compute_histogram(layer_weights)
    bias_hist, bias_bins = compute_histogram(layer_biases)

    # Plot weight distribution
    plt.hist(layer_weights, bins=30, alpha=0.5, label='Weights')
    plt.hist(layer_biases, bins=30, alpha=0.5, label='Biases')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Weight and Bias Distribution for {layer_name}')
    plt.legend()
    plt.show()

    # Plot KDE for weights
    sns.kdeplot(layer_weights, label='Weights')
    sns.kdeplot(layer_biases, label='Biases')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Weight and Bias KDE for {layer_name}')
    plt.legend()
    plt.show()

    # Compute statistics
    weight_mean = np.mean(layer_weights)
    weight_std = np.std(layer_weights)
    weight_min = np.min(layer_weights)
    weight_max = np.max(layer_weights)
    near_zero_threshold = 0.01
    near_zero_fraction = np.mean(np.abs(layer_weights) < near_zero_threshold)

    print(f"Statistics for {layer_name}:")
    print(f"  Weights - Mean: {weight_mean:.4f}, Std: {weight_std:.4f}, Min: {weight_min:.4f}, Max: {weight_max:.4f}")
    print(f"  Fraction of near-zero weights: {near_zero_fraction:.4f}")

# Layer-wise correlation heatmap
layer_names = list(weights.keys())
weight_means = [np.mean(weights[layer].flatten()) for layer in layer_names]
weight_stds = [np.std(weights[layer].flatten()) for layer in layer_names]

# Create a DataFrame for correlation analysis
import pandas as pd
data = {'Layer': layer_names, 'Mean Weight': weight_means, 'Std Weight': weight_stds}
df = pd.DataFrame(data)

# Compute correlation matrix
corr_matrix = df[['Mean Weight', 'Std Weight']].corr()

# Plot correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Between Mean and Std of Weights Across Layers')
plt.show()

# 3D Surface Plot for Weight Distributions (if you have epoch-wise data)
# Assuming you have epoch-wise weight data (replace with actual data)
epochs = range(50)  # Example: 50 epochs
for layer_name in weights.keys():
    # Simulate epoch-wise weight data (replace with actual data)
    weight_data = np.random.randn(len(epochs), 100)  # Example: 100 weights per epoch

    # Smooth the data
    from scipy.ndimage import gaussian_filter
    Z_smoothed = gaussian_filter(weight_data, sigma=1)

    # Create 3D plot
    X, Y = np.meshgrid(np.arange(weight_data.shape[1]), epochs)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_smoothed, cmap='viridis', alpha=1, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel('Weight Index')
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Weight Value')
    ax.set_title(f'Weight Distribution Over Epochs for {layer_name}')
    plt.show()

# Compare initial vs final weight distributions (if you have epoch-wise data)
initial_weights = weight_data[0, :]
final_weights = weight_data[-1, :]
plt.hist(initial_weights, bins=30, alpha=0.5, label='Initial Weights')
plt.hist(final_weights, bins=30, alpha=0.5, label='Final Weights')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Initial vs Final Weight Distribution')
plt.legend()
plt.show()