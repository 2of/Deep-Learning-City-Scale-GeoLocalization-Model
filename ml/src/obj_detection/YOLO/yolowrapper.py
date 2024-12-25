from ultralytics import YOLO


class YOLOWrapper:
    def __init__(self, model_path):
        """
        Initialize the YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model weights file.
        """
        self.model = YOLO(model_path)
        print(f"YOLO model loaded from {model_path}")

        # Retrieve class names from the model
        self.class_names = self.model.names

    def predict(self, image_path):
        results = self.model.predict(image_path)
        return results


    def get_label_names(self, labels):
        label_names = [self.class_names[int(label)] for label in labels]
        return label_names

# Example usage
if __name__ == "__main__":
    # Initialize the wrapper with the model path
    yolo_wrapper = YOLOWrapper('yolo11n.pt')
    
    # Run inference on an image
    image_path = './sample_data/samplestreetviews/chicago.png'
    results, labels = yolo_wrapper.predict(image_path)
    
    # Get the label names
    label_names = yolo_wrapper.get_label_names(labels)
    print("Detected label names:", label_names)