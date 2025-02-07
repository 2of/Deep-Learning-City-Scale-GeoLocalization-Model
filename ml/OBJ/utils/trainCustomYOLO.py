from ultralytics import YOLO
import torch

# Enable CUDA Launch Blocking for better error messages
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'


model = YOLO('yolo11n.pt')  # load a pretrained model

# Print model summary
print(model)

try:
    results = model.train(data='./datasets/yolo2/onlytrafficlights/data.yaml', epochs=50)  # train the model with your dataset
except RuntimeError as e:
    print(f"RuntimeError during training: {e}")

model.save('TLIGHTS.pt')

print("Model training complete and saved as 'obj_detection_1.pt'.")