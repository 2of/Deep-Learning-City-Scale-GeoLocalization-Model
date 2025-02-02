from ultralytics import YOLO
import cv2
import sys
import torch

class YOLO_OBJ_Wrapper:
    def __init__(self, model_path, conf_threshold=0.001, filters=None, class_mapping=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.filters = filters
        self.class_mapping = class_mapping or {}
        print(f"YOLO (OBJ) model loaded from {model_path} with filters on classes {filters}")
        self.class_names = self.model.names
        # self.print_class_names()  # Print class names upon initialization to verify

    def predict(self, image, andshow=False, and_print_labels=False):
        # Perform prediction
        results = self.model(image)
        
        # Filter results based on confidence threshold and class filters
        filtered_results = []
        class_ids_vector = []
        confidences = []
        bboxes = []
        
        if self.filters:
            for result in results:
                filtered_boxes = [box for box in result.boxes if box.conf[0] >= self.conf_threshold and box.cls[0] in self.filters]
                if filtered_boxes:
                    result.boxes = filtered_boxes
                    filtered_results.append(result)
                    for box in filtered_boxes:
                        class_id = int(box.cls[0])
                        mapped_class_id = self.class_mapping.get(class_id, class_id)
                        class_name = self.class_names[mapped_class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # Correctly extract the bounding box coordinates
                        confidences.append(confidence)
                        bboxes.append(bbox)
                        class_ids_vector.append(mapped_class_id)
                        if and_print_labels:
                            print(f"Class ID: {class_id} (Mapped: {mapped_class_id}), Class Name: {class_name}, Confidence: {confidence}, BBox: {bbox}")  # Debug print
        else:
            for result in results:
                filtered_boxes = [box for box in result.boxes if box.conf[0] >= self.conf_threshold]
                if filtered_boxes:
                    result.boxes = filtered_boxes
                    filtered_results.append(result)
                    for box in filtered_boxes:
                        class_id = int(box.cls[0])
                        mapped_class_id = self.class_mapping.get(class_id, class_id)
                        class_name = self.class_names[mapped_class_id]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # Correctly extract the bounding box coordinates
                        confidences.append(confidence)
                        bboxes.append(bbox)
                        class_ids_vector.append(mapped_class_id)
                        if and_print_labels:
                            print(f"Class ID: {class_id} (Mapped: {mapped_class_id}), Class Name: {class_name}, Confidence: {confidence}, BBox: {bbox}")  # Debug print
        
        if andshow:
            for result in filtered_results:
                result.show()
        
        # Create tensor for class IDs vector
        tensor_class_ids_vector = torch.tensor(class_ids_vector, dtype=torch.int32)

        # Create tensor for bounding boxes
        tensor_bboxes = torch.tensor(bboxes, dtype=torch.float32)

        # Create tensor for confidence scores
        tensor_confidences = torch.tensor(confidences, dtype=torch.float32)

        return tensor_class_ids_vector, tensor_bboxes, tensor_confidences

    def print_class_names(self):
        print("Class names for the model:")
        for idx, class_name in enumerate(self.class_names):
            print(f"{idx}: {class_name}")

if __name__ == "__main__":
    model_path = "./models/yolo11n.pt"  
    image_path = "./sample_data/samplestreetviews/chicago.png"  # Replace with your sample image path

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        sys.exit(1)

    # Initialize YOLOWrapper with class mapping
    TLIGHTS_ETC_CLASSES = {9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 74: 7, 58: 8}
    yolo_wrapper = YOLO_OBJ_Wrapper(model_path, filters=[9, 10, 11, 12, 13, 74], class_mapping=TLIGHTS_ETC_CLASSES)

    # Print all possible labels for the model
    yolo_wrapper.print_class_names()

    # Predict and plot
    tensor_class_ids_vector, tensor_bboxes, tensor_confidences = yolo_wrapper.predict(image, andshow=True, and_print_labels=True)
    print("Tensor of Class IDs Vector:")
    print(tensor_class_ids_vector)
    print("Bounding Boxes:")
    print(tensor_bboxes)
    print("Confidence Scores:")
    print(tensor_confidences)