import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_hsv_histogram(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the histogram for each channel
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Normalize the histograms
    h_hist = cv2.normalize(h_hist, h_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(s_hist, s_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(v_hist, v_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).flatten()

    # Plot the histograms
    plt.figure(figsize=(12, 6))

    # Plot Hue histogram
    plt.plot(h_hist, color='r', label='Hue')

    # Plot Saturation histogram
    plt.plot(s_hist, color='g', label='Saturation')

    # Plot Value histogram
    plt.plot(v_hist, color='b', label='Value')

    plt.title('HSV Histogram')
    plt.xlabel('Bin')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
plot_hsv_histogram('HISTIMAGE.png')
