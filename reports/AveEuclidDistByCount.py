import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'OVERALL_NO_ATTN_WITH_STATS/preds_test_vs_actual.csv'
data = pd.read_csv(file_path)

# Compute Euclidean distance for each row
data['Euclidean_Distance'] = np.sqrt(
    (data['Predicted Latitude'] - data['True Latitude'])**2 +
    (data['Predicted Longitude'] - data['True Longitude'])**2
)

# Parameters
num_samples = 18
num_points = 512
mse_values = np.zeros((num_samples, num_points))  # Store all samples

# Compute MSE across multiple samples
for i in range(num_samples):
    sample_data = data.sample(n=num_points, replace=False)

    for n in range(1, num_points + 1):
        mse_values[i, n - 1] = np.mean(sample_data['Euclidean_Distance'].iloc[:n]**2)

# Compute mean and standard deviation
mse_mean = np.mean(mse_values, axis=0)
mse_std = np.std(mse_values, axis=0)

# Plot results
plt.figure(figsize=(10, 6), dpi=120)
x_vals = range(1, num_points + 1)

# Main line: Mean MSE
plt.plot(x_vals, mse_mean, linestyle='-', color='b', label="Mean Averaged MSE")

# Shaded area: ±1 standard deviation (68% confidence interval)
plt.fill_between(x_vals, mse_mean - mse_std, mse_mean + mse_std, color='b', alpha=0.2, label="±1 Std Dev")

plt.xlabel('Number of Rows Averaged', fontsize=12)
plt.ylabel('Averaged MSE', fontsize=12)
plt.title('Averaged MSE For Averaged Record Length', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
