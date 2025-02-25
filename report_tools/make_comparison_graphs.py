import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directories
dir_no_attn = 'OVERALL_NO_ATTN_WITH_STATS/oneper'
dir_attn = 'OVERALL_NO_ATTN_WITH_STATS/oneper'

# Define the file suffixes
suffixes = ['LAT', 'LONG', 'OVERALL']

# Plot each pair of files (no attention vs attention) for each suffix
for suffix in suffixes:
    plt.figure(figsize=(10, 6))
    
    # Read and plot the no attention file
    file_no_attn = os.path.join(dir_no_attn, f'test_mses_{suffix}.csv')
    df_no_attn = pd.read_csv(file_no_attn)
    plt.plot(df_no_attn, label=f'No Attention Test {suffix}')
    
    # Read and plot the attention file
    file_attn = os.path.join(dir_attn, f'test_mses_{suffix}.csv')
    df_attn = pd.read_csv(file_attn)
    plt.plot(df_attn, label=f'Attention Test {suffix}')
    
    plt.title(f'Comparison of Test MSEs for {suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

# Plot the OVERALL values for train and test in the same directory on the same graph
plt.figure(figsize=(10, 6))

# No Attention: Train and Test OVERALL
file_no_attn_train = os.path.join(dir_no_attn, 'train_mses_OVERALL.csv')
file_no_attn_test = os.path.join(dir_no_attn, 'test_mses_OVERALL.csv')
df_no_attn_train = pd.read_csv(file_no_attn_train)
df_no_attn_test = pd.read_csv(file_no_attn_test)
plt.plot(df_no_attn_train, label='No Attention Train OVERALL')
plt.plot(df_no_attn_test, label='No Attention Test OVERALL')

plt.title('No Attention: Train vs Test OVERALL MSEs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))

# Attention: Train and Test OVERALL
file_attn_train = os.path.join(dir_attn, 'train_mses_OVERALL.csv')
file_attn_test = os.path.join(dir_attn, 'test_mses_OVERALL.csv')
df_attn_train = pd.read_csv(file_attn_train)
df_attn_test = pd.read_csv(file_attn_test)
plt.plot(df_attn_train, label='Attention Train OVERALL')
plt.plot(df_attn_test, label='Attention Test OVERALL')

plt.title('Attention: Train vs Test OVERALL MSEs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()