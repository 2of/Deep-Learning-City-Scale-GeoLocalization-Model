import os
import random
import math
import tensorflow as tf

'''manually split up data'''

# PATHS!
src_dir = './data/tfrecords/fivetwelve/src'
train_dir = './data/tfrecords/fivetwelve/train'
test_dir = './data/tfrecords/fivetwelve/test'
val_dir = './data/tfrecords/fivetwelve/val'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of all TFRecord files
tfrecord_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.tfrecord')]

# Shuffle the files
random.shuffle(tfrecord_files)

# Split the files into train, test, and val sets
num_files = len(tfrecord_files)
num_val_files = math.ceil(0.1 * num_files)
num_test_files = math.ceil(0.1 * num_files)
num_train_files = num_files - num_val_files - num_test_files

train_files = tfrecord_files[:num_train_files]
val_files = tfrecord_files[num_train_files:num_train_files + num_val_files]
test_files = tfrecord_files[num_train_files + num_val_files:]

def write_tfrecord(files, output_dir):
    total_records = 0
    for file in files:
        output_path = os.path.join(output_dir, os.path.basename(file))
        with tf.io.TFRecordWriter(output_path) as writer:
            for record in tf.data.TFRecordDataset(file):
                writer.write(record.numpy())
                total_records += 1
    print(f"Written {total_records} records to {output_dir}")
    return total_records

# Write the TFRecord files to their respective directories
print("Writing validation TFRecord files...")
val_records_written = write_tfrecord(val_files, val_dir)

print("Writing test TFRecord files...")
test_records_written = write_tfrecord(test_files, test_dir)

print("Writing train TFRecord files...")
train_records_written = write_tfrecord(train_files, train_dir)

# Verify that the total number of records matches the original, with a leeway of three records
original_total_records = sum([sum(1 for _ in tf.data.TFRecordDataset(file)) for file in tfrecord_files])
new_total_records = val_records_written + test_records_written + train_records_written

print(f"Original total records: {original_total_records}")
print(f"New total records: {new_total_records}")

if abs(new_total_records - original_total_records) <= 3:
    print("TFRecord files have been successfully split and saved.")
else:
    print("Error: Mismatch in total number of records. No changes have been made to the original files.")

# Print the total number of rows in each set
total_train_rows = sum([sum(1 for _ in tf.data.TFRecordDataset(os.path.join(train_dir, f))) for f in os.listdir(train_dir) if f.endswith('.tfrecord')])
total_val_rows = sum([sum(1 for _ in tf.data.TFRecordDataset(os.path.join(val_dir, f))) for f in os.listdir(val_dir) if f.endswith('.tfrecord')])
total_test_rows = sum([sum(1 for _ in tf.data.TFRecordDataset(os.path.join(test_dir, f))) for f in os.listdir(test_dir) if f.endswith('.tfrecord')])

print(f"Total rows in train set: {total_train_rows}")
print(f"Total rows in validation set: {total_val_rows}")
print(f"Total rows in test set: {total_test_rows}")

print("TFRecord files have been split and saved.")