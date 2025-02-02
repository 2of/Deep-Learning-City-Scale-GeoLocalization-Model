import pandas as pd

def filter_dataframe_by_ids(pickle_file, missing_ids_csv, output_pickle_file):
    # Load the missing IDs from the CSV file
    missing_ids_df = pd.read_csv(missing_ids_csv)
    missing_ids = set(missing_ids_df['missing_id'])
    print(f"Loaded {len(missing_ids)} missing IDs from {missing_ids_csv}")

    # Load the original DataFrame from the pickle file
    df = pd.read_pickle(pickle_file)
    print(f"Loaded DataFrame from {pickle_file} with {len(df)} rows")

    # Filter the DataFrame to keep only the rows with the missing IDs
    filtered_df = df[df['id'].isin(missing_ids)]
    print(f"Filtered DataFrame to {len(filtered_df)} rows with missing IDs")

    # Set the 'is_retrieved' field to False for all rows
    filtered_df['is_retrieved'] = False
    print(f"Set 'is_retrieved' field to False for all rows in the filtered DataFrame")

    # Save the filtered DataFrame to a new pickle file
    filtered_df.to_pickle(output_pickle_file)
    print(f"Saved filtered DataFrame to {output_pickle_file}")

if __name__ == "__main__":
    pickle_file = "./chicago.pkl"
    missing_ids_csv = "./missing_ids.csv"
    output_pickle_file = "./filtered_chicago.pkl"
    
    filter_dataframe_by_ids(pickle_file, missing_ids_csv, output_pickle_file)