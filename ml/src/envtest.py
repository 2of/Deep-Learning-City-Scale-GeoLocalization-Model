import tensorflow as tf

def test_tensorflow():
    try:
        print("TensorFlow Version:", tf.__version__)
        # Test TensorFlow operations
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        print("TensorFlow computation test passed:", c.numpy())

        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPUs detected: {len(gpus)}")
            for gpu in gpus:
                print(" -", gpu.name)
        else:
            print("No GPU detected. Running on CPU.")
    except Exception as e:
        print("TensorFlow test failed.")
        print(e)

if __name__ == "__main__":
    test_tensorflow()