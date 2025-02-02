from ultralytics import YOLO
import torch

# Load your model
model = YOLO("./models/yolo11n.pt")

# Perform a dummy prediction to get the class names
dummy_image = torch.zeros((1, 3, 640, 640))  # Create a dummy image tensor
results = model(dummy_image)
print(results)
# Extract class names from the results
class_names = results.names

# Print all class names and their corresponding indices
print("Class names for the model:")
for idx, class_name in enumerate(class_names):
    print(f"{idx}: {class_name}")
