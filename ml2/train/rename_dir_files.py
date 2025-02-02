# Only exists because my linux dist file manager doesnt support 'merge and rename'... 


import os

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.tfrecord') and filename.startswith('output_'):
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_A{ext}"
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            print(f"Renamed {old_file} to {new_file}")

# Example usage
directory = './data/tfrecords/MAIN_BRANCH_TRAIN_VAL'
rename_files_in_directory(directory)