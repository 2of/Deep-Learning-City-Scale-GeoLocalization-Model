from ultralytics import YOLO
import torch
import numpy as np

class YoloWrapper:
    def __init__(self, model_path="yolo11n.pt", device="cpu"):
        self.device = device
        if device == "cpu":
            print("!!!!!!!!!!!!!!!!!!!!!!!!!! ")
            print("!!! RUNNING ON CPU !!!!!!")
            print("Large batch size will kill arm macOS ")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.model = YOLO(model_path).to(device)

    def train(self):
        print("Training functionality will be implemented here.")
        pass

    def evaluate(self):
        """
        Placeholder for evaluating the YOLO model.
        """
        print("Evaluation functionality will be implemented here.")
        pass

    def predict(self, input_tensor):
        print("CALLED PREDICT WITH : ")
        print(input_tensor.shape)

        """
        Runs object detection on a batch of input tensors.
        Args:
            input_tensor (torch.Tensor or np.ndarray): 
                Tensor of shape (batch_size, 640, 640, 3) with normalized pixel values (0 to 1).
        Returns:
            list of torch.Tensor: Bounding boxes (xyxy), class IDs, and confidence scores.
        """
        # If input is a NumPy array, convert it to a PyTorch tensor
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor)

        # Ensure the input tensor is on the correct device
        input_tensor = input_tensor.to(self.device)

        # Reshape tensor if necessary (batch dimension must be first)
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if needed

        # YOLO expects shape (batch_size, 3, 640, 640), so permute if necessary
        if input_tensor.shape[-1] == 3:
            input_tensor = input_tensor.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)

        # Run inference using the YOLO model
        results = self.model.predict(source=input_tensor, device=self.device)

        # Extract relevant outputs (bounding boxes, class IDs, confidence scores)
        output_tensors = []
        for result in results:
            boxes = result.boxes.xyxy  # Bounding boxes in (x1, y1, x2, y2) format
            class_ids = result.boxes.cls  # Class IDs
            confidences = result.boxes.conf  # Confidence scores

            # Create a dictionary for easy access
            output_tensors.append({
                "boxes": boxes,
                "class_ids": class_ids,
                "confidences": confidences
            })

        return output_tensors
    
    
# a = YoloWrapper()
