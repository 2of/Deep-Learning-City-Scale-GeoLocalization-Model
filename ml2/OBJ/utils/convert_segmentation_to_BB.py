import os
from PIL import Image

def convert_labels_to_bounding_boxes(directory):
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(directory, split, 'images')
        labels_dir = os.path.join(directory, split, 'labels')
        output_dir = os.path.join(directory, split, 'bounding_boxes')

        if not os.path.exists(labels_dir):
            print(f"Labels directory {labels_dir} does not exist.")
            continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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

            bounding_boxes = []
            for label in labels:
                parts = label.strip().split()
                if len(parts) != 5:
                    print(f"Invalid label format in file {label_path}: {label}")
                    continue

                class_label, x_center, y_center, bbox_width, bbox_height = map(float, parts)

                # Convert normalized coordinates to pixel values
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                x_max = (x_center + bbox_width / 2) * width
                y_max = (y_center + bbox_height / 2) * height

                bounding_boxes.append((class_label, x_min, y_min, x_max, y_max))

            # Save bounding boxes to a new file
            output_file = os.path.join(output_dir, label_file)
            with open(output_file, 'w') as f:
                for bbox in bounding_boxes:
                    f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
convert_labels_to_bounding_boxes('datasets/yolodetection/Geoinformer_training_data/')