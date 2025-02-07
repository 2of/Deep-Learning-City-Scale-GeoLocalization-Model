import os
import tensorflow as tf

def get_num_records_in_tfrecord(file_path):
    count = 0
    for _ in tf.data.TFRecordDataset(file_path):
        count += 1
    return count

def get_total_num_records_in_dir(directory):
    total_count = 0
    num_files = 0
    for filename in os.listdir(directory):
        if filename.endswith(".tfrecord"):
            file_path = os.path.join(directory, filename)
            total_count += get_num_records_in_tfrecord(file_path)
            num_files += 1
    return total_count, num_files

if __name__ == "__main__":
    directory = "./data/tfrecords/fivetwelve/train"
    total_records, num_files = get_total_num_records_in_dir(directory)
    print(f"Total number of records in all .tfrecord files: {total_records}")
    print(f"Total number of .tfrecord files: {num_files}")