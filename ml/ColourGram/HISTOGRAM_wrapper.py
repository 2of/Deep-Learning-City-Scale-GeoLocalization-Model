import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import torch


class HISTOGRAM_WRAPPER:
    def __init__(self):
        pass
        


    def compute_histogram(self, image):
        """
        Compute the color histogram of an image and normalize it.
        
        Args:
            image (numpy.ndarray): The input image.
        
        Returns:
            dict: A dictionary containing normalized histograms for each color channel.
        """
        # Convert image to RGB if it's not already
        if image.shape[2] == 4:  # If the image has an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # Compute the histogram !!
        histograms = {}
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms[col] = hist / hist.sum()  # Normalize the histogram
        
        return histograms

    def plot_histogram(self, histograms):
        """
        Plot the color histogram.
        
        Args:
            histograms (dict): A dictionary containing histograms for each color channel.
        """
        for col, hist in histograms.items():
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()
        
        
    def normalize_histograms(self,histograms):
        return histograms / histograms.sum(dim=1, keepdim=True)
    def flatten_histograms(self,histograms):
        return histograms.view(histograms.size(0), -1)

    def create_histogram_embedding(self,images, bins=256):
        histograms = self.get_color_histogram_tensor_stack(images, bins)
        normalized_histograms = self.normalize_histograms(histograms)
        flattened_histograms = self.flatten_histograms(normalized_histograms)
        return flattened_histograms

    #?????? Without it, no matter WHAT gets passed to <images/> its always an instance of the class? 
    # Absolutely no idea why....... (neither does chatgpt!!)
    @staticmethod
    def get_color_histogram_tensor_stack(images, bins=64):
        if images is None:
            raise ValueError("Input tensor is None")
        # Ensure the input tensor is of type float
        # print(type(images))
        images = images.float()
        
        # images: Tensor of shape (N, 3, 128, 128)
        N, C, H, W = images.shape
        histograms = []

        for i in range(N):
            histogram = []
            for c in range(C):
                hist = torch.histc(images[i, c], bins=bins, min=0, max=1)
                histogram.append(hist)
            histograms.append(torch.cat(histogram))

        return torch.stack(histograms)

    def get_color_histogram_tensor_from_single_image(self, image):
        """
        Get the color histogram of an image as an embedding in a tensor.
        
        Args:
            image (PIL.Image.Image or str): The input image or path to the input image.
        
        Returns:
            tf.Tensor: A tensor containing the color histogram embedding.
        """
        # Check if the input is a path or a PIL Image object
        if isinstance(image, str):
            # Load the image from the path
            image = Image.open(image)
        
        # Convert PIL Image to numpy array
        image = np.array(image)
        
        # Compute the histogram
        histograms = self.compute_histogram(image)
        
        # Concatenate histograms into a single array
        histogram_array = np.concatenate([histograms['b'], histograms['g'], histograms['r']]).flatten()
        
        # Convert the array to a tensor
        histogram_tensor = tf.convert_to_tensor(histogram_array, dtype=tf.float32)
        
        return histogram_tensor



    def get_color_histogram_tensor(self, image):
        """
        Get the color histogram of an image as an embedding in a tensor.
        
        Args:
            image (PIL.Image.Image or str): The input image or path to the input image.
        
        Returns:
            tf.Tensor: A tensor containing the color histogram embedding.
        """
        # Check if the input is a path or a PIL Image object
        if isinstance(image, str):
            # Load the image from the path
            image = Image.open(image)
        
        # Convert PIL Image to numpy array
        image = np.array(image)
        
        # Compute the histogram
        histograms = self.compute_histogram(image)
        
        # Concatenate histograms into a single array
        histogram_array = np.concatenate([histograms['b'], histograms['g'], histograms['r']]).flatten()
        
        # Convert the array to a tensor
        histogram_tensor = tf.convert_to_tensor(histogram_array, dtype=tf.float32)
        
        return histogram_tensor

# if __name__ == "__main__":
#     histogram_wrapper = HISTOGRAM_WRAPPER()
#     image_path = 'path_to_your_image.jpg'
#     histogram_tensor = histogram_wrapper.get_color_histogram_tensor(image_path)
    
#     print("Histogram Tensor:", histogram_tensor)