import os
import cv2
import numpy as np
from glob import glob

class YOLOv11Dataset:
    def __init__(self, dataset_path):
        self.image_paths = glob(os.path.join(dataset_path, "images", "*.jpg"))
        self.label_paths = {os.path.basename(p).replace(".jpg", ".txt"): p for p in glob(os.path.join(dataset_path, "labels", "*.txt"))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        label_path = self.label_paths.get(os.path.basename(img_path).replace(".jpg", ".txt"))
        labels = self.load_labels(label_path)
        return image, labels

    def load_labels(self, label_path):
        labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    labels.append([float(x) for x in line.strip().split()])
        return np.array(labels)

class ImageAugmentor:
    def __init__(self, dataset_path, output_path, augmentations):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.augmentations = augmentations
        self.dataset = YOLOv11Dataset(dataset_path)

    def apply_augmentations(self, image, labels):
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        for aug in self.augmentations:
            augmented_image, augmented_labels = aug(augmented_image, augmented_labels)
        return augmented_image, augmented_labels

    def save_image(self, image, path):
        cv2.imwrite(path, image)

    def save_labels(self, labels, path):
        with open(path, "w") as file:
            for label in labels:
                file.write(" ".join(map(str, label)) + "\n")

    def process_dataset(self):
        if not os.path.exists(os.path.join(self.output_path, "images")):
            os.makedirs(os.path.join(self.output_path, "images"))
        if not os.path.exists(os.path.join(self.output_path, "labels")):
            os.makedirs(os.path.join(self.output_path, "labels"))

        for img_path in self.dataset.image_paths:
            image, labels = self.dataset[self.dataset.image_paths.index(img_path)]
            augmented_image, augmented_labels = self.apply_augmentations(image, labels)
            output_img_path = os.path.join(self.output_path, "images", os.path.basename(img_path))
            output_label_path = os.path.join(self.output_path, "labels", os.path.basename(img_path).replace(".jpg", ".txt"))
            self.save_image(augmented_image, output_img_path)
            self.save_labels(augmented_labels, output_label_path)

def random_flip(image, labels):
    flipped_image = cv2.flip(image, 1)
    h, w = image.shape[:2]
    flipped_labels = labels.copy()
    flipped_labels[:, 1] = 1 - labels[:, 1]  # Flip x_center
    return flipped_image, flipped_labels

def random_rotate(image, labels):
    angle = np.random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    rotated_labels = labels.copy()
    for i, label in enumerate(labels):
        x_center, y_center = label[1] * w, label[2] * h
        new_x_center = (M[0, 0] * x_center + M[0, 1] * y_center + M[0, 2]) / w
        new_y_center = (M[1, 0] * x_center + M[1, 1] * y_center + M[1, 2]) / h
        rotated_labels[i, 1] = new_x_center
        rotated_labels[i, 2] = new_y_center
    return rotated_image, rotated_labels

def random_brightness(image, labels):
    value = np.random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * value
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image, labels
import os
import cv2
import numpy as np
from glob import glob

class YOLOv11Dataset:
    def __init__(self, dataset_path):
        self.image_paths = glob(os.path.join(dataset_path, "images", "*.jpg"))
        self.label_paths = {os.path.basename(p).replace(".jpg", ".txt"): p for p in glob(os.path.join(dataset_path, "labels", "*.txt"))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        label_path = self.label_paths.get(os.path.basename(img_path).replace(".jpg", ".txt"))
        labels = self.load_labels(label_path)
        return image, labels

    def load_labels(self, label_path):
        labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    labels.append([float(x) for x in line.strip().split()])
        return np.array(labels)

class ImageAugmentor:
    def __init__(self, dataset_path, output_path, augmentations):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.augmentations = augmentations
        self.dataset = YOLOv11Dataset(dataset_path)

    def apply_augmentations(self, image, labels):
        augmented_image = image.copy()
        augmented_labels = labels.copy()
        for aug in self.augmentations:
            augmented_image, augmented_labels = aug(augmented_image, augmented_labels)
        return augmented_image, augmented_labels

    def save_image(self, image, path):
        cv2.imwrite(path, image)

    def save_labels(self, labels, path):
        with open(path, "w") as file:
            for label in labels:
                file.write(" ".join(map(str, label)) + "\n")

    def process_dataset(self):
        if not os.path.exists(os.path.join(self.output_path, "images")):
            os.makedirs(os.path.join(self.output_path, "images"))
        if not os.path.exists(os.path.join(self.output_path, "labels")):
            os.makedirs(os.path.join(self.output_path, "labels"))

        for img_path in self.dataset.image_paths:
            image, labels = self.dataset[self.dataset.image_paths.index(img_path)]
            augmented_image, augmented_labels = self.apply_augmentations(image, labels)
            output_img_path = os.path.join(self.output_path, "images", os.path.basename(img_path))
            output_label_path = os.path.join(self.output_path, "labels", os.path.basename(img_path).replace(".jpg", ".txt"))
            self.save_image(augmented_image, output_img_path)
            self.save_labels(augmented_labels, output_label_path)

def random_flip(image, labels):
    flipped_image = cv2.flip(image, 1)
    h, w = image.shape[:2]
    flipped_labels = labels.copy()
    flipped_labels[:, 1] = 1 - labels[:, 1]  # Flip x_center
    return flipped_image, flipped_labels

def random_rotate(image, labels):
    angle = np.random.uniform(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    rotated_labels = labels.copy()
    for i, label in enumerate(labels):
        x_center, y_center = label[1] * w, label[2] * h
        new_x_center = (M[0, 0] * x_center + M[0, 1] * y_center + M[0, 2]) / w
        new_y_center = (M[1, 0] * x_center + M[1, 1] * y_center + M[1, 2]) / h
        rotated_labels[i, 1] = new_x_center
        rotated_labels[i, 2] = new_y_center
    return rotated_image, rotated_labels

def random_brightness(image, labels):
    value = np.random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * value
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image, labels

def process_all_datasets(base_path, output_base_path, augmentations):
    for subset in ['train', 'valid', 'test']:
        dataset_path = os.path.join(base_path, subset)
        output_path = os.path.join(output_base_path, subset)
        augmentor = ImageAugmentor(dataset_path, output_path, augmentations)
        augmentor.process_dataset()

if __name__ == "__main__":
    base_path = "./datasets/yolo2/test_for_augmnet"
    output_base_path = "./datasets/yolo2/test_for_augmnet/again"
    augmentations = [random_flip, random_rotate, random_brightness]

    process_all_datasets(base_path, output_base_path, augmentations)
