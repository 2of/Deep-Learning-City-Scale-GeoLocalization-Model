import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
base_path = 'OVERALL_NO_ATTN_WITH_STATS'
epochs = range(50)
layers = ['batch_normalization_1', 'batch_normalization_2', 'dense_1', 'dense_2', 'dense_3', 'dense_4', 'dense_5', 'dense']

data = {layer: {'hist': [], 'bins': []} for layer in layers}

for epoch in epochs:
    for layer in layers:
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

# Compute bin centers (assuming bins are the same for all epochs and layers)
bin_centers = (data[layers[0]]['bins'][0][:-1] + data[layers[0]]['bins'][0][1:]) / 2

# Compute mean weight values for each layer across epochs
layer_mean_data = []
for layer in layers:
    hist = data[layer]['hist']
    mean_weights = np.sum(hist * bin_centers, axis=1) / np.sum(hist, axis=1)
    layer_mean_data.append(mean_weights)

# Debug: Print the mean weight values for each layer
for layer, mean_data in zip(layers, layer_mean_data):
    print(f"{layer}: {mean_data}")

# Compute correlation matrix
layer_correlations = np.corrcoef(layer_mean_data)

# Debug: Print the correlation matrix
print("Correlation Matrix:")
print(layer_correlations)

# Plot the correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(layer_correlations, annot=True, xticklabels=[layer.replace("_", " ").title() for layer in layers], yticklabels=[layer.replace("_", " ").title() for layer in layers])
plt.title('Correlation Between Layer Weight Distributions')
plt.show()