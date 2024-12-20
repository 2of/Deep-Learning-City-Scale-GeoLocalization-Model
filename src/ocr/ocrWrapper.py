import tensorflow as tf
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import numpy as np

class OCRModel:
    def __init__(self, pretrained=True, assume_straight_pages=True, export_as_straight_boxes=True):
        # Load the pre-trained OCR model with the specified options
        self.model = ocr_predictor(pretrained=pretrained, assume_straight_pages=assume_straight_pages, export_as_straight_boxes=export_as_straight_boxes)

    def predict(self, input_tensor):
        # Ensure the tensor is in the right format (height, width, channels)
        if len(input_tensor.shape) != 3 or input_tensor.shape[2] != 3:
            raise ValueError("Input tensor must have the shape [height, width, 3]")

        # Convert numpy array to a TensorFlow tensor if it isn't already
        if not isinstance(input_tensor, tf.Tensor):
            input_tensor = tf.convert_to_tensor(input_tensor)

        # Use the model to predict on the tensor
        result = self.model([input_tensor])

        return result

# Example usage:
if __name__ == "__main__":
    # Create an instance of the OCRModel
    ocr_model = OCRModel(pretrained=True, assume_straight_pages=True, export_as_straight_boxes=True)

    # Example: Load your tensor (assuming it's a numpy array here)
    # Let's create a dummy tensor for this example
    input_tensor = np.random.rand(256, 256, 3).astype(np.float32)  # Replace with your actual tensor

    # Predict using the OCR model
    result = ocr_model.predict(input_tensor)

    # The result is a doctr Document object
    print(result)

    # You can visualize or further process the result
    result.show()

    # Rebuild the original document from its predictions
    import matplotlib.pyplot as plt

    synthetic_pages = result.synthesize()
    plt.imshow(synthetic_pages[0])
    plt.axis('off')
    plt.show()
