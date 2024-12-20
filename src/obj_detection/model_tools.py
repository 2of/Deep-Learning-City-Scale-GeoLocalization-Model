
import torch
import onnx
from ultralytics import YOLO
from onnx_tf.backend import prepare
import os





class YoloModelConverter:
    def __init__(self, pytorch_model_path, output_dir):
        self.pytorch_model_path = pytorch_model_path
        self.output_dir = output_dir
        self.onnx_model_path = os.path.join(output_dir, "yolo_model.onnx")
        self.tf_model_path = os.path.join(output_dir, "yolo_tf_model")

    def convert(self):
        """
        Convert the PyTorch YOLO model to TensorFlow via ONNX.
        """

        model = YOLO(self.pytorch_model_path)

        # export to onnx
        print("export PyTorch model to ONNX...")
        model.export(format="onnx", path=self.onnx_model_path)


        print("load the ONNX model...")
        onnx_model = onnx.load(self.onnx_model_path)

        # onvert ONNX model to TensorFlow
        print("convert ONNX model to TensorFlow...")
        tf_rep = prepare(onnx_model)

        # Step 5: Save the TensorFlow model
        print(f"Saving the TensorFlow model to {self.tf_model_path}...")
        tf_rep.export_graph(self.tf_model_path)

        print("Conversion complete.")


pytorch_model_path = "path_to_your_pytorch_model/yolo11n.pt"
output_dir = "path_to_save_converted_model"

converter = YoloModelConverter(pytorch_model_path, output_dir)
converter.convert()