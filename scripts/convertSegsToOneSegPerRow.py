import os
import tensorflow as tf
import argparse

'''
This code is generated by AI ~~
Essentially, we take the records that are of the form 
{lat, long : {text embeddings , , , , ....... [6] }, {colour histograms , , , , .......[6]}}

and make 

[6] records with the same lat,long but with each the appropriate text and colour embeddings

(because for training we want random distribution of the signs, not of the sources of the signs as those are 
sorta linear already in their relationship, we want the groupings 'learnt' from the model. not to be a product of ordering 

/rant also this one loads everything to memory so it's slow..... 
)
'''

def parse_tf_example(example_proto):
    # should really pop this in utils.py
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

def process_tfrecords(input_dir, output_file):
    tfrecord_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tfrecord')]
    total_files = len(tfrecord_files)
    print(f"Found {total_files} TFRecord files in {input_dir}")

    total_records_written = 0

    with tf.io.TFRecordWriter(output_file) as writer:
        for tfrecord_file in tfrecord_files:
            print(f"Processing file: {tfrecord_file}")
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            parsed_dataset = raw_dataset.map(parse_tf_example)

            for parsed_record in parsed_dataset:
                latitude = parsed_record['latitude'].numpy()
                longitude = parsed_record['longitude'].numpy()
                text_embeddings = tf.sparse.to_dense(parsed_record['text_embeddings']).numpy()
                color_histograms = tf.sparse.to_dense(parsed_record['color_histograms']).numpy()
                num_detections = parsed_record['num_detections'].numpy()

                print(f"Record: lat={latitude}, lon={longitude}, num_detections={num_detections}")
                print(f"text_embeddings shape: {text_embeddings.shape}")
                print(f"color_histograms shape: {color_histograms.shape}")

                # Ensure text embeddings are reshaped to match num_detections
                text_embeddings = text_embeddings.reshape((num_detections, 12, 128))
                print(f"Reshaped text_embeddings shape: {text_embeddings.shape}")

                # Ensure color histograms are reshaped to match num_detections
                color_histograms = color_histograms.reshape((num_detections, 256))
                print(f"Reshaped color_histograms shape: {color_histograms.shape}")

                # Create a record for each detection
                for i in range(num_detections):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'latitude': tf.train.Feature(float_list=tf.train.FloatList(value=[latitude])),
                        'longitude': tf.train.Feature(float_list=tf.train.FloatList(value=[longitude])),
                        'text_embeddings': tf.train.Feature(float_list=tf.train.FloatList(value=text_embeddings[i].flatten())),
                        'color_histograms': tf.train.Feature(float_list=tf.train.FloatList(value=color_histograms[i].flatten()))
                    }))
                    writer.write(example.SerializeToString())
                    total_records_written += 1
                    print(f"Written detection {i+1}/{num_detections} for record with lat={latitude}, lon={longitude}")

    print(f"Processed {total_files} TFRecord files and wrote {total_records_written} records to {output_file}")

def main():
    input_dir = "./data/tfrecords"
    output_file = "./data/tfrecords/singlerows/combined.tfrecord"
    process_tfrecords(input_dir, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and flatten TFRecord files.")
    parser.add_argument("--input_dir", type=str, default="./data/tfrecords/fivetwelve/src", help="Directory containing the TFRecord files to process")
    parser.add_argument("--output_file", type=str, default="./data/tfrecords/onesegperrow/combined.tfrecord", help="File to save the processed TFRecord data")
    args = parser.parse_args()

    process_tfrecords(args.input_dir, args.output_file)