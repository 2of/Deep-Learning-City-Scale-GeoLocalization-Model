import tensorflow as tf
import os
import argparse

'''
Short tool to merge a bunch of TFRecord files into multiple TFRecord files,
each containing the combined records of a specified number of input files.
'''

def parse_tf_example(example_proto):
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'text': tf.io.VarLenFeature(tf.string),
        'text_embeddings': tf.io.VarLenFeature(tf.float32),
        'color_histograms': tf.io.VarLenFeature(tf.float32),
        'num_detections': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def combine_tfrecords(input_dir, output_dir, files_per_batch):
    tfrecord_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tfrecord')]
    total_files = len(tfrecord_files)
    print(f"Found {total_files} TFRecord files in {input_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grand_total_rows = 0

    for i in range(0, total_files, files_per_batch):
        batch_files = tfrecord_files[i:i + files_per_batch]
        output_file = os.path.join(output_dir, f"combined_{i // files_per_batch + 1}.tfrecord")
        total_rows = 0
        with tf.io.TFRecordWriter(output_file) as writer:
            for tfrecord_file in batch_files:
                raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
                row_count = sum(1 for _ in raw_dataset)
                print(f"{tfrecord_file} with {row_count} rows")
                total_rows += row_count
                for raw_record in raw_dataset:
                    writer.write(raw_record.numpy())
        grand_total_rows += total_rows
        print(f"Combined {len(batch_files)} TFRecord files into {output_file} with {total_rows} rows")

    print(f"Total number of rows written across all new files: {grand_total_rows}")

def main():
    input_dir = "./data/tfrecords"
    output_dir = "./data/tfrecords/combined"
    files_per_batch = 5
    combine_tfrecords(input_dir, output_dir, files_per_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple TFRecord files into batches.")
    parser.add_argument("--input_dir", type=str, default="./data/tfrecords/allbatches", help="Directory containing the TFRecord files to combine")
    parser.add_argument("--output_dir", type=str, default="./data/tfrecords/batched2/", help="Directory to save the combined TFRecord files")
    parser.add_argument("--files_per_batch", type=int, default=10, help="Number of TFRecord files to combine into one")
    args = parser.parse_args()

    combine_tfrecords(args.input_dir, args.output_dir, args.files_per_batch)