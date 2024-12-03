import torch
import onnx
from onnx_tf.backend import prepare
import argparse
import os

def convert_model_pyt_to_tf(source_model, output_location):
    # Load the PyTorch model
    model = torch.load(source_model)
    model.eval()

    # Export to ONNX format
    onnx_path = "model.onnx"
    dummy_input = torch.randn(1, 3, 640, 640)  # Assuming the model takes 640x640 RGB input
    torch.onnx.export(model, dummy_input, onnx_path)

    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert to TensorFlow using ONNX-TF
    tf_rep = prepare(onnx_model)

    # Export the TensorFlow model
    tf_rep.export_graph(output_location)

    print(f"Model successfully converted to TensorFlow and saved at {output_location}")

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to TensorFlow format")
    parser.add_argument("source_model", type=str, help="Path to the PyTorch model (.pt or .pth file)")
    parser.add_argument("output_location", type=str, help="Directory to save the converted TensorFlow model")

    # Parse arguments
    args = parser.parse_args()

    # Check if the source model file exists
    if not os.path.exists(args.source_model):
        print(f"Error: The source model '{args.source_model}' does not exist.")
    else:
        # Call the conversion function
        convert_model_pyt_to_tf(args.source_model, args.output_location)