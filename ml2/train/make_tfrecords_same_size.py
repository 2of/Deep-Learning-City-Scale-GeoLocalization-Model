import tensorflow as tf
import os

def create_fixed_size_tfrecords(input_dir, output_dir, fixed_size=1024):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _parse_function(proto):
        feature_description = {
            'id': tf.io.FixedLenFeature([], tf.int64),
            'latitude': tf.io.FixedLenFeature([], tf.float32),
            'longitude': tf.io.FixedLenFeature([], tf.float32),
            'text': tf.io.VarLenFeature(tf.string),
            'text_embeddings': tf.io.VarLenFeature(tf.float32),
            'color_histograms': tf.io.VarLenFeature(tf.float32),
            'num_detections': tf.io.FixedLenFeature([], tf.int64)
        }
        return tf.io.parse_single_example(proto, feature_description)

    def serialize_example(parsed_record):
        feature = {
            'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_record['id'].numpy()])),
            'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[parsed_record['latitude'].numpy()])),
            'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[parsed_record['longitude'].numpy()])),
            'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.numpy() for t in tf.sparse.to_dense(parsed_record['text'])])),
            'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=tf.sparse.to_dense(parsed_record['text_embeddings']).numpy())),
            'color_histograms': tf.train.Feature(float_list=tf.train.FloatList(value=tf.sparse.to_dense(parsed_record['color_histograms']).numpy())),
            'num_detections': tf.train.Feature(int64_list=tf.train.Int64List(value=[parsed_record['num_detections'].numpy()]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def write_tfrecords(records, output_path):
        with tf.io.TFRecordWriter(output_path) as writer:
            for record in records:
                writer.write(record)

    buffer = []
    file_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.tfrecord'):
            input_path = os.path.join(input_dir, filename)
            raw_dataset = tf.data.TFRecordDataset(input_path)
            parsed_dataset = raw_dataset.map(_parse_function)

            for parsed_record in parsed_dataset:
                try:
                    serialized_record = serialize_example(parsed_record)
                    buffer.append(serialized_record)
                except Exception as e:
                    print(f"Error serializing record from file {filename}: {e}")
                    continue

                if len(buffer) == fixed_size:
                    output_path = os.path.join(output_dir, f"fixed_size_{file_count}.tfrecord")
                    write_tfrecords(buffer, output_path)
                    buffer = []
                    file_count += 1

    if buffer:
        output_path = os.path.join(output_dir, f"fixed_size_{file_count}.tfrecord")
        write_tfrecords(buffer, output_path)

if __name__ == "__main__":
    input_directory = './data/tfrecords/batched2/train'
    output_directory = './data/tfrecords/fixedsize/train_fixed_size'
    create_fixed_size_tfrecords(input_directory, output_directory)