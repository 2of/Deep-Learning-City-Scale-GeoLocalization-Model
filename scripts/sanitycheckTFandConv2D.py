import tensorflow as tf

'''
Imagine a beautiful, lovign world where pip3 install tensorflow and all the rest creates a perfect, harmonious world of compatiability.
this file just makes fake model to determine of is working. 

was just loaded as chat gpt code 

'''

def test_conv2d_layer():
    # Create a simple Conv2D layer
    conv_layer = tf.keras.layers.Conv2D(
        filters=32, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same', 
        activation='relu'
    )

    # Create a random input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal([1, 28, 28, 3])

    # Pass the input tensor through the Conv2D layer
    output_tensor = conv_layer(input_tensor)

    # Print the shape of the output tensor
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

if __name__ == "__main__":
    test_conv2d_layer()