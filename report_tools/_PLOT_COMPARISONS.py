import matplotlib.pyplot as plt

# Function to read a file with one value per line
def read_values(file_path):
    with open(file_path, 'r') as file:
        values = [float(line.strip()) for line in file if line.strip()]
    return values

# Load the data from the files
val_mses = read_values('OVERALL_ATTN_WITH_STATS/oneper/val_mses_OVERALL.csv')
train_mses = read_values('OVERALL_ATTN_WITH_STATS/oneper/train_mses_OVERALL.csv')
test_mses = read_values('OVERALL_ATTN_WITH_STATS/oneper/test_mses_OVERALL.csv')



# Load the data from the files
other_val_mses = read_values('OVERALL_NO_ATTN_WITH_STATS/oneper/val_mses_OVERALL.csv')
other_train_mses = read_values('OVERALL_NO_ATTN_WITH_STATS/oneper/train_mses_OVERALL.csv')
othertest_mses = read_values('OVERALL_NO_ATTN_WITH_STATS/oneper/test_mses_OVERALL.csv')

# Generate x-axis values (epochs or steps) based on the length of each dataset
val_epochs = range(1, len(val_mses) + 1)
train_epochs = range(1, len(train_mses) + 1)
test_epochs = range(1, len(test_mses) + 1)

# Plot the data on the same axis
plt.figure(figsize=(10, 6))

plt.plot(val_epochs, val_mses, label='Validation MSE')
plt.plot(train_epochs, train_mses, label='Training MSE')
plt.plot(test_epochs, test_mses, label='Test MSE')

# Add labels and title
plt.xlabel('Epoch/Step')
plt.ylabel('MSE')
plt.title('Training, Validation, and Test MSE over Epochs/Steps')
plt.legend()

# Show the plot
plt.show()