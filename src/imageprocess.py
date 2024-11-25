import tensorflow as tf

def compute_hsv_histogram_batch(image_batch):
    """
    Computes the HSV histogram for a batch of images.
    
    Args:
        image_batch (tf.Tensor): Batch of image tensors with shape (batch_size, height, width, channels).
        
    Returns:
        histograms (tf.Tensor): Batch of histograms with HSV values.
    """
    # Convert the image batch from RGB to HSV
    hsv_batch = tf.image.rgb_to_hsv(image_batch)

    # Initialize lists to store histograms for each channel
    histograms = []

    for i in range(hsv_batch.shape[0]):
        # Get the HSV channels for the current image
        h, s, v = hsv_batch[i, :, :, 0], hsv_batch[i, :, :, 1], hsv_batch[i, :, :, 2]
        
        # Compute histograms for each channel
        hist_h = tf.histogram_fixed_width(h, [0.0, 1.0], nbins=256)
        hist_s = tf.histogram_fixed_width(s, [0.0, 1.0], nbins=256)
        hist_v = tf.histogram_fixed_width(v, [0.0, 1.0], nbins=256)
        
        # Normalize histograms
        hist_h = hist_h / tf.reduce_sum(hist_h)
        hist_s = hist_s / tf.reduce_sum(hist_s)
        hist_v = hist_v / tf.reduce_sum(hist_v)
        
        # Combine histograms into a single tensor
        hist = tf.stack([hist_h, hist_s, hist_v], axis=0)
        
        histograms.append(hist)

    # Combine all histograms into a single tensor
    histograms = tf.stack(histograms, axis=0)

    return histograms

# Example usage:
# Assuming batch of image tensors with shape (batch_size, 640, 640, 3)
