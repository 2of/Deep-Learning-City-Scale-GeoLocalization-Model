import fiftyone as fo

# Set MongoDB connection string
fo.config.database_uri = "mongodb://localhost:27017"



# Load the Open Images V7 validation split with specific classes and label types
dataset = fo.zoo.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections", "segmentations"],
    classes=["Street Sign", "Traffic Light"],  # Replace with relevant classes
    max_samples=100,  # Adjust sample size as needed
)

# Print dataset information
print(dataset)


# Launch the FiftyOne App to view the dataset
session = fo.launch_app(dataset)



import matplotlib.pyplot as plt
import fiftyone.utils.plot as fop

# Visualize the first 5 images with their labels
for sample in dataset.take(5):
    # Display the image with its detections
    fop.draw_labeled_image(sample, figsize=(12, 8))
    plt.show()