import fiftyone as fo

# Set MongoDB connection string (already in your code)
fo.config.database_uri = "mongodb://localhost:27017"

# Load a dataset
dataset = fo.zoo.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=["Street Sign", "Traffic Light"],  # Replace with your classes
    max_samples=100,
)

# Launch the app
session = fo.launch_app(dataset, port=5151)
