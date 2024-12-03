##############################################
#                                            #
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   ~         🚀 WELCOME TO THE TEST AREA!   #
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                            #
#        🏗️  🚧  PLAY WITH CODE HERE  🚧  🏗️  #
#                                            #
#   --------------------------------------   #
#   |     WARNING: Chaos Ahead! 🚨         |  #
#   |     Debuggers Welcome 😎             |   #
#   |     Coordinates May Vary 📍          |   #
#   --------------------------------------    #
#                                            #
#       🛰️   📦  ⚙️   🗺️   🏙️   🔄   📦        #
#                                            #
#     Prototype → Crash → Repeat!            #
#                                            #
#        🌍     🏞️    🏛️    🏢     🌋          #
#                                            #
##############################################



# This file is just for validation and messing about, esnurign crap works
# Load in images:


import sys
import os
import tensorflow as tf
current_directory = os.getcwd()
sys.path.append(os.path.join(current_directory, 'src'))

from ingest.inputWrapper import inputWrapper
# Import the loss functions from modeltools.loss
from modeltools.loss import haversine_loss_batch, haversine_loss_single
from colour.hsv_to_tensor import hsv_histogram_batch, convert_to_embedding
from obj_detection.yolowrapper import YoloWrapper


csv_path = os.path.join(current_directory, 'res/samplestreetviews/index.csv')
image_dir = os.path.join(current_directory, 'res/samplestreetviews')

image_data = inputWrapper(csv_path, image_dir)
image_data.load_all_auto()

sample_batch,sample_labels = image_data.load_batch(image_dir,batch_num=1,batch_size=3)
sample_labels = tf.convert_to_tensor(sample_labels, dtype=tf.float32)
print(type(sample_batch))
print(sample_batch.shape,sample_labels[0:2])

## Hot dog! We have code loaded for the tensors, god I despise python directory nonsense and macos permissions nonsense...


## SAMPLE SAMPLE PASS PASS ## SAMPLE SAMPLE PASS PASS ##

#oh boy 
print("SHAPE OF INITIAL TENSOR")
print(sample_batch.shape)
hist_tensor = hsv_histogram_batch(sample_batch)
print("SHAPE OF HSV'd TENSOR")
print(hist_tensor.shape)

embedding = convert_to_embedding(hist_tensor, embedding_dim=128)
print("SHAPE OF embedd'd TENSOR")
print(embedding.shape)

print(embedding)
print("now let's run some OCR on one of those tensors")

