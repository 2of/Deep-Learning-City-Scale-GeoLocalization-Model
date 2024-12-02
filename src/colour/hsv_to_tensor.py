import tensorflow as tf

def hsv_histogram_batch(image_batch):
    # Convert the image batch from RGB to HSV
    hsv_batch = tf.image.rgb_to_hsv(image_batch)
    histograms = []

    for i in range(hsv_batch.shape[0]):
        # Get the HSV channels for particular iamge
        h, s, v = hsv_batch[i, :, :, 0], hsv_batch[i, :, :, 1], hsv_batch[i, :, :, 2]
        
        # Compute histograms for each channel
        hist_h = tf.histogram_fixed_width(h, [0.0, 1.0], nbins=256)
        hist_s = tf.histogram_fixed_width(s, [0.0, 1.0], nbins=256)
        hist_v = tf.histogram_fixed_width(v, [0.0, 1.0], nbins=256)
        
        # Normalize 
        hist_h = hist_h / tf.reduce_sum(hist_h)
        hist_s = hist_s / tf.reduce_sum(hist_s)
        hist_v = hist_v / tf.reduce_sum(hist_v)
        
        # Combine histograms to single tensor
        hist = tf.stack([hist_h, hist_s, hist_v], axis=0)
        # add to hists
        histograms.append(hist)

    # Combine all histograms into a single tensor
    histograms = tf.stack(histograms, axis=0)

    return histograms


def convert_to_embedding(histogram_tensor, embedding_dim=128):
    """
    Convert the output of hsv_histogram_batch into a useful embedding.
    
    Parameters:

    
    Returns:

    """

    flat_histogram = tf.reshape(histogram_tensor, [histogram_tensor.shape[0], -1])

    #flatten for stability
    flat_histogram = tf.keras.layers.LayerNormalization()(flat_histogram)

    # Project into a dense layer for learned embedding
    embedding = tf.keras.layers.Dense(embedding_dim, activation='relu')(flat_histogram)

    # TODO change?
    embedding = tf.keras.layers.Dropout(0.2)(embedding)

    return embedding