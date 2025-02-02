import pickle
import random
import pandas as pd

# Path to the pickle file
pickle_file_path = './chicago.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    df = pickle.load(file)

print(df)
# Choose 2 random indexes from the DataFrame
random_indexes = random.sample(range(len(df)), 2)

# Print the selected indexes and their corresponding rows to the console
for index in random_indexes:
    print(f'Index: {index}, Row: {df.iloc[index]}')