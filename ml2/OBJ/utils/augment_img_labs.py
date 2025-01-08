import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# Function to load annotations from a YOLO format text file
def load_annotations(annotation_path):
    bboxes = []
    category_ids = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            category_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
            category_ids.append(category_id)
    return bboxes, category_ids

# Function to apply augmentation and visualize bounding boxes
def augment_and_visualize(image_path, annotation_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotations
    bboxes, category_ids = load_annotations(annotation_path)
    
    # Convert YOLO format to Pascal VOC format
    height, width, _ = image.shape
    pascal_voc_bboxes = []
    for bbox in bboxes:
        x_center, y_center, w, h = bbox
        x_min = int((x_center - w / 2) * width)
        y_min = int((y_center - h / 2) * height)
        x_max = int((x_center + w / 2) * width)
        y_max = int((y_center + h / 2) * height)
        pascal_voc_bboxes.append([x_min, y_min, x_max, y_max])
    
    # Apply augmentation
    augmented = transform(image=image, bboxes=pascal_voc_bboxes, category_ids=category_ids)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    
    # Visualize the augmented image with bounding boxes
    for bbox in augmented_bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(augmented_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(augmented_image)
    plt.axis('off')
    plt.show()
    
    # Uncomment the following lines to save the augmented images
    # output_dir = 'augmented_images'
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, os.path.basename(image_path))
    # cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

# Hardcoded paths for sample image and annotation from the train set
sample_image_path = './datasets/Signs/train/sample_image.jpg'
sample_annotation_path = './datasets/Signs/train/sample_image.txt'

augment_and_visualize(sample_image_path, sample_annotation_path)
