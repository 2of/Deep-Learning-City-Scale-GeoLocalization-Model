import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directories
dir_attn = 'OVERALL_ATTN_WITH_STATS'
dir_no_attn = 'OVERALL_NO_ATTN_WITH_STATS'

# Function to safely load and extract the first row
def load_first_row(file_path):
    df = pd.read_csv(file_path)
    if not df.empty:
        return df.iloc[0]
    else:
        return pd.Series()  # Return an empty Series if the DataFrame is empty

# Load test MSEs
test_mses_attn_lat = load_first_row(os.path.join(dir_attn, 'test_mses_LAT.csv'))
test_mses_attn_long = load_first_row(os.path.join(dir_attn, 'test_mses_LONG.csv'))
test_mses_attn_overall = load_first_row(os.path.join(dir_attn, 'test_mses_OVERALL.csv'))

test_mses_no_attn_lat = load_first_row(os.path.join(dir_no_attn, 'test_mses_LAT.csv'))
test_mses_no_attn_long = load_first_row(os.path.join(dir_no_attn, 'test_mses_LONG.csv'))
test_mses_no_attn_overall = load_first_row(os.path.join(dir_no_attn, 'test_mses_OVERALL.csv'))

# Load train and validation MSEs
def load_mses(directory, prefix):
    lat = pd.read_csv(os.path.join(directory, f'{prefix}_mses_LAT.csv'))
    long = pd.read_csv(os.path.join(directory, f'{prefix}_mses_LONG.csv'))
    overall = pd.read_csv(os.path.join(directory, f'{prefix}_mses_OVERALL.csv'))
    return lat, long, overall

train_mses_attn_lat, train_mses_attn_long, train_mses_attn_overall = load_mses(dir_attn, 'train')
val_mses_attn_lat, val_mses_attn_long, val_mses_attn_overall = load_mses(dir_attn, 'val')

train_mses_no_attn_lat, train_mses_no_attn_long, train_mses_no_attn_overall = load_mses(dir_no_attn, 'train')
val_mses_no_attn_lat, val_mses_no_attn_long, val_mses_no_attn_overall = load_mses(dir_no_attn, 'val')

# Aggregate train and validation MSEs by epoch (mean across batches)
def aggregate_mses(mses):
    return mses.mean(axis=1)

train_mses_attn_lat_agg = aggregate_mses(train_mses_attn_lat)
train_mses_attn_long_agg = aggregate_mses(train_mses_attn_long)
train_mses_attn_overall_agg = aggregate_mses(train_mses_attn_overall)

val_mses_attn_lat_agg = aggregate_mses(val_mses_attn_lat)
val_mses_attn_long_agg = aggregate_mses(val_mses_attn_long)
val_mses_attn_overall_agg = aggregate_mses(val_mses_attn_overall)

train_mses_no_attn_lat_agg = aggregate_mses(train_mses_no_attn_lat)
train_mses_no_attn_long_agg = aggregate_mses(train_mses_no_attn_long)
train_mses_no_attn_overall_agg = aggregate_mses(train_mses_no_attn_overall)

val_mses_no_attn_lat_agg = aggregate_mses(val_mses_no_attn_lat)
val_mses_no_attn_long_agg = aggregate_mses(val_mses_no_attn_long)
val_mses_no_attn_overall_agg = aggregate_mses(val_mses_no_attn_overall)

# Plotting
sns.set(style="whitegrid")

# Test MSE Comparison
plt.figure(figsize=(12, 6))
plt.bar(['LAT', 'LONG', 'OVERALL'], [test_mses_attn_lat.mean(), test_mses_attn_long.mean(), test_mses_attn_overall.mean()], label='With Attention', alpha=0.7)
plt.bar(['LAT', 'LONG', 'OVERALL'], [test_mses_no_attn_lat.mean(), test_mses_no_attn_long.mean(), test_mses_no_attn_overall.mean()], label='Without Attention', alpha=0.7)
plt.title('Test MSE Comparison')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Train and Validation MSE Comparison
def plot_mses(mses_attn, mses_no_attn, title):
    plt.figure(figsize=(12, 6))
    plt.plot(mses_attn, label='With Attention')
    plt.plot(mses_no_attn, label='Without Attention')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

plot_mses(train_mses_attn_lat_agg, train_mses_no_attn_lat_agg, 'Train MSE Comparison (LAT)')
plot_mses(train_mses_attn_long_agg, train_mses_no_attn_long_agg, 'Train MSE Comparison (LONG)')
plot_mses(train_mses_attn_overall_agg, train_mses_no_attn_overall_agg, 'Train MSE Comparison (OVERALL)')

plot_mses(val_mses_attn_lat_agg, val_mses_no_attn_lat_agg, 'Validation MSE Comparison (LAT)')
plot_mses(val_mses_attn_long_agg, val_mses_no_attn_long_agg, 'Validation MSE Comparison (LONG)')
plot_mses(val_mses_attn_overall_agg, val_mses_no_attn_overall_agg, 'Validation MSE Comparison (OVERALL)')