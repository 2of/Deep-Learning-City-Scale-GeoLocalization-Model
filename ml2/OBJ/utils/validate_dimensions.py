import os
from PIL import Image

def find_invalid_files(directory):
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(directory, split, 'images')
        labels_dir = os.path.join(directory, split, 'labels')

        if not os.path.exists(labels_dir):
            print(f"Labels directory {labels_dir} does not exist.")
            continue

        for label_file in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, label_file)
            image_path = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))  # Assuming images are in .jpg format

            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist.")
                continue

            with open(label_path, 'r') as f:
                labels = f.readlines()

            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            for label in labels:
                parts = label.strip().split()
                if len(parts) < 5:
                    print(f"Invalid label format in file {label_path}: {label}")
                    break

                class_label, x_center, y_center, bbox_width, bbox_height = map(float, parts[:5])

                # Convert normalized coordinates to pixel values
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                x_max = (x_center + bbox_width / 2) * width
                y_max = (y_center + bbox_height / 2) * height

                if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                    print(f"Invalid bounding box in file {label_path}: {label}")
                    print(f"Image file: {image_path}")
                    break

find_invalid_files('datasets/yolodetection/Geoinformer_training_data/')