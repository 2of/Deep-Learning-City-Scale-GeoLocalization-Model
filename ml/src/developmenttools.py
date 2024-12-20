from loadimages import *
from imageprocess import *
# Example usage:
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../res/samplestreetviews'))
batch_size = 2
dataset = load_images(image_dir, batch_size)



for batch in dataset: 
    print ("STOCK")
    print(batch.shape)
    print ("HSV")
    a = compute_hsv_histogram_batch(batch)
    print(a.shape)