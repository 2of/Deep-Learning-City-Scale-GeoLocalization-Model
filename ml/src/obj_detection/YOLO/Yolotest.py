from ultralytics import YOLO

# Load a COCO-pretrained YOLOv11 model
model = YOLO('yolo11n.pt')

# Path to your image
image_path = 'res/samplestreetviews/chicago.png'

# Run inference
results = model.predict(image_path)

# Display results
for result in results:
    result.show()