import csv
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = 'MSE_COMPARISON.csv'

# Read the CSV file using the csv module
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Check if the CSV has exactly two rows
if len(data) != 2:
    raise ValueError("The CSV file must contain exactly two rows for comparison.")

# Extract the two rows and convert them to floats
row1 = [float(value) for value in data[0]]
row2 = [float(value) for value in data[1]]

# Optionally, you could apply log transformation to deal with large differences in scale
row1_log = np.log(np.array(row1) + 1e-10)  # Adding a small constant to avoid log(0)
row2_log = np.log(np.array(row2) + 1e-10)

# Plot the two rows against each other (before and after log transformation)
plt.figure(figsize=(10, 6))

# Plot the original rows (raw values)
plt.plot(row1, label='Row 1 (Raw)', color='blue', linestyle='-', marker='o')
plt.plot(row2, label='Row 2 (Raw)', color='red', linestyle='-', marker='x')

# Optionally, plot log-transformed rows if magnitude difference is large
plt.plot(row1_log, label='Row 1 (Log)', color='blue', linestyle='--', marker='o')
plt.plot(row2_log, label='Row 2 (Log)', color='red', linestyle='--', marker='x')

# Customize the plot
plt.title('Comparison of Two Rows in MSE_COMPARISON.csv (Raw vs Log Transformed)')
plt.xlabel('Columns')
plt.ylabel('Values (Raw & Log Scale)')
plt.legend()
plt.grid(True)
plt.show()