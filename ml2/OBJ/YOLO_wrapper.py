import numpy as np
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import sys
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import torch
import torchvision.transforms as T

class YOLOWrapper:
    def __init__(self, model_path,conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print(f"YOLO model loaded from {model_path}")
        self.class_names = self.model.names
        
        self.labels = ["dog", "banana", "horse"] #sample
    def predict(self, image, andshow=False, andshowfromBB=True):
        transform = T.Compose([T.ToTensor()]) 
        image_tensor = transform(image).unsqueeze(0)
        
        # Suppress print statements from self.model
        sys.stdout = open(os.devnull, 'w')
        results = self.model(image_tensor)
        sys.stdout = sys.__stdout__

        bounding_boxes, labels = self.get_bounding_boxes_from_results(results)

        if andshow:
            for result in results:
                result.show()

        if andshowfromBB:
            for box in bounding_boxes:
                cropped_image = self.cut_out_boundingbox(image_tensor.squeeze(0), box)
                padded_image = self.pad_and_resize(cropped_image, (128, 128))
                plt.imshow(padded_image.permute(1, 2, 0).cpu().numpy())
                plt.show()

        return results

    def predict_single_image(self, image):
        test = self.model.predict(image)
        print(test)
        return test
        
        
        
        
        
    def get_objects_and_labels(self, image, id=123):
        transform = T.Compose([T.ToTensor()]) 
        image_tensor = transform(image).unsqueeze(0)

        # Ensure the input shape is divisible by stride 32!!!~
        _, _, h, w = image_tensor.shape
        stride = 32
        if h % stride != 0 or w % stride != 0:
            new_h = ((h // stride) + 1) * stride
            new_w = ((w // stride) + 1) * stride
            resize_transform = T.Resize((new_h, new_w))
            image_tensor = resize_transform(image_tensor)

        # results = self.model(image_tensor)
        results = self.model(image_tensor, conf=self.conf_threshold)
        # Print the number of items detected
        num_detections = len(results[0].boxes)
        # print(f"Number of items detected: {num_detections}")
        if num_detections == 0: 
            return (None, None, None)
        # Get bounding boxes and labels from YOLO model
        bounding_boxes, labels = self.get_bounding_boxes_from_results(results)

        # Initialize tensor to store the padded images
        padded_images_tensor = torch.stack([self.pad_and_resize(self.cut_out_boundingbox(image_tensor.squeeze(0), box), (128, 128)) for box in bounding_boxes])

        # Convert labels to a tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Calculate the center points of each bounding box
        centers = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)) for box in bounding_boxes]

        # Compute the distance matrix
        distance_matrix = self.calculate_distance_matrix(centers)

        return padded_images_tensor, labels_tensor, distance_matrix

    def get_bounding_boxes_from_results(self, results):
        bounding_boxes = []
        labels = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0]
                bounding_boxes.append(coords[:4])
                labels.append(box.cls)
        return bounding_boxes, labels

    def cut_out_boundingbox(self, image, bb_coords):
        # print(type(image))
        x1, y1, x2, y2 = bb_coords
        cropped_image = image[:, int(y1):int(y2), int(x1):int(x2)]
        return cropped_image

    def pad_and_resize(self,image, size):
        c, h, w = image.shape
        target_h, target_w = size

        # Calculate padding
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = F.resize(image, (new_h, new_w))

        pad_w = target_w - new_w
        pad_h = target_h - new_h

        # Create an alpha channel with the same size as the resized image
        alpha_channel = torch.ones((1, new_h, new_w))

        # Concatenate the alpha channel to the resized image
        resized_image_with_alpha = torch.cat((resized_image, alpha_channel), dim=0)

        # Pad the image with the alpha channel
        padded_image_with_alpha = F.pad(resized_image_with_alpha, (0, 0, pad_w, pad_h), fill=0)

        return padded_image_with_alpha
    
    def calculate_distance_matrix(self, centers):
        num_detections = len(centers)
        distance_matrix = np.zeros((num_detections, num_detections), dtype=float)

        for i in range(num_detections):
            for j in range(num_detections):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))

        return distance_matrix


        
# # Example usage
# if __name__ == "__main__":
#     yolo_wrapper = YOLOWrapper('yolo11n.pt')
#     image_path = './sample_data/samplestreetviews/chicago.png'
#     results = yolo_wrapper.predict(image_path)
#     print(results.shape)
#     # readable_output = yolo_wrapper.get_readable_output(
#     #     results["embedding"], results["distance_matrix"]
#     # )
#     # print("Detected label features and relationships:\n", readable_output)
