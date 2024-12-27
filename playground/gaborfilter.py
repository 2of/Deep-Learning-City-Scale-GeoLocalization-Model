import cv2
import numpy as np
from skimage.filters import gabor
import matplotlib.pyplot as plt

# Load the image
image_path = './sample_data/samplestreetviews/chicago.png'  # Make sure to replace this with the correct path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Unable to load image. Please check the file path.")
else:
    # Define Gabor filter parameters
    frequencies = [0.05, 0.25]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # Apply Gabor filters
    gabor_kernels = []
    for theta in thetas:
        for frequency in frequencies:
            kernel = np.real(gabor(image, frequency=frequency, theta=theta)[0])
            gabor_kernels.append(kernel)

    # Display Gabor filter results
    fig, axs = plt.subplots(1, len(gabor_kernels), figsize=(20, 5))
    for i, kernel in enumerate(gabor_kernels):
        axs[i].imshow(kernel, cmap='gray')
        axs[i].set_title(f'Gabor {i+1}')
    plt.show()