import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dropout, BatchNormalization
import tensorflow as tf

import tensorflow as tf

    
    
    
    
# class concatModelForSegments(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         # Define the layers for the model
#         self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
#         self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(256, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(128, activation='relu')
#         self.output_layer = tf.keras.layers.Dense(2, activation='linear')  # Output layer for latitude and longitude

#     def call(self, inputs, mask=None):
#         # print("call with inputs: ")
#         # print(inputs)
#         # print("\n")
        
#         text_embeddings = inputs['text_embeddings']
#         color_histograms = inputs['color_histograms']

#         # # Ensure the shapes are as expected
#         # if len(text_embeddings.shape) != 3 or text_embeddings.shape[1:] != (12, 128):
#         #     raise ValueError(f"Unexpected shape for text_embeddings: {text_embeddings.shape}")
#         # if len(color_histograms.shape) != 2 or color_histograms.shape[1] != 256:
#         #     raise ValueError(f"Unexpected shape for color_histograms: {color_histograms.shape}")

#         # Reshape color_histograms to (batch_size, 2, 128)
#         color_histograms_reshaped = tf.reshape(color_histograms, (-1, 2, 128))
#         # print("color_histograms_reshaped: ", color_histograms_reshaped.shape)

#         # Concatenate along the second axis
#         concatenated_inputs = tf.concat([text_embeddings, color_histograms_reshaped], axis=1)
#         # print("concatenated_inputs: ", concatenated_inputs.shape)

#         # Expand dimensions to match Conv2D input requirements
#         concatenated_inputs = tf.expand_dims(concatenated_inputs, -1)
#         # print("expanded concatenated_inputs: ", concatenated_inputs.shape)

#         # sooooooo 
#         # BASIC for now. 
#         # TO TRAIN TONIGHT (watch me forget!!!!)
#         x = self.conv1(concatenated_inputs)
#         # print("after conv1: ", x.shape)
#         x = self.conv2(x)
#         # print("after conv2: ", x.shape)
#         x = self.flatten(x)
#         # print("after flatten: ", x.shape)
#         x = self.dense1(x)
#         # print("after dense1: ", x.shape)
#         x = self.dense2(x)
#         # print("after dense2: ", x.shape)
#         x = self.output_layer(x)
#         # print("after output_layer: ", x.shape)

#         return x
    


class concatModelForSegments_attn(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers for the model
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='linear')  # Output layer for latitude and longitude

        # Attention layer
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)

    def call(self, inputs, mask=None):
        text_embeddings = inputs['text_embeddings']
        color_histograms = inputs['color_histograms']

        # Print shapes for debugging
        #print("text_embeddings shape:", text_embeddings.shape)
       # print("color_histograms shape:", color_histograms.shape)
#
        # Attention mechanism on text_embeddings
        attention_output, attention_weights = self.multi_head_attention(text_embeddings, text_embeddings, return_attention_scores=True)
       # print("attention_output shape:", attention_output.shape)
       # print("attention_weights shape:", attention_weights.shape)

        # Reshape color_histograms to (batch_size, 2, 128)
        color_histograms_reshaped = tf.reshape(color_histograms, (-1, 2, 128))
     #   print("color_histograms_reshaped shape:", color_histograms_reshaped.shape)

        # Concatenate along the second axis
        concatenated_inputs = tf.concat([attention_output, color_histograms_reshaped], axis=1)
       # print("concatenated_inputs shape:", concatenated_inputs.shape)

        # Expand dimensions to match Conv2D input requirements
        concatenated_inputs = tf.expand_dims(concatenated_inputs, -1)
        #print("expanded concatenated_inputs shape:", concatenated_inputs.shape)

        x = self.conv1(concatenated_inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)

        return x
    
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, query, value):
        return self.attention(query=query, value=value)

class fullimageModel2(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__()
        # Define the layers for the model
        self.class_name_embedding = tf.keras.layers.Embedding(input_dim=8, output_dim=128)  # Adjust input_dim as needed
        self.text_embedding_dense = tf.keras.layers.Dense(128, activation='elu')
        self.color_histogram_dense = tf.keras.layers.Dense(128, activation='elu')
        
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)  # Adjust num_heads and key_dim as needed
        self.concat_dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.concat_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.concat_dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='linear')  # Output layer for latitude and longitude

        # Regularization
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.l2_regularizer = tf.keras.regularizers.l2(0.01)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, mask=None):
        text_embeddings = inputs['text_embeddings']
        color_histograms = inputs['color_histograms']
        class_names = inputs['class_names_vectors']
        bboxes = inputs["bboxes"]
        
        # Pad class_names to a fixed size
        class_names = tf.pad(class_names, [[0, 0], [0, 11 - tf.shape(class_names)[1]]])

        # Convert class_names to one-hot encoding
        class_names_one_hot = tf.one_hot(class_names, depth=1000)  # Adjust depth as needed
        
        # Create embeddings for one-hot encoded class names
        class_name_embeddings = tf.reduce_sum(self.class_name_embedding(class_names_one_hot), axis=2)
        
        # Reshape to (batch_size, 128, 11)
        class_name_embeddings = tf.transpose(class_name_embeddings, perm=[0, 2, 1])
        
        # Apply dense layers to each feature
        text_embeddings = self.text_embedding_dense(text_embeddings)
        color_histograms = self.color_histogram_dense(color_histograms)
        
        # Combine text embeddings and class name embeddings for attention
        text_embeddings = tf.transpose(text_embeddings, perm=[0, 2, 1])  # Reshape to (batch_size, 128, 12)
        combined_embeddings = tf.concat([text_embeddings, class_name_embeddings], axis=2)
        
        # Ensure combined_embeddings shape is compatible with MultiHeadAttention
        combined_embeddings = tf.reshape(combined_embeddings, [combined_embeddings.shape[0], -1, self.multi_head_attention.key_dim])
        
        # Apply multi-head attention mechanism
        attended_embeddings = self.multi_head_attention(query=combined_embeddings, value=combined_embeddings)
        attended_embeddings = tf.reshape(attended_embeddings, [attended_embeddings.shape[0], 128, -1])
        
        color_histograms = tf.reshape(color_histograms, [color_histograms.shape[0], 128, -1])  # (batch_size, 128, 3)
        
        # Concatenate all features along the 128 dimension
        combined_features = tf.concat([attended_embeddings, color_histograms], axis=2)
        
        # Flatten the combined features
        combined_features = tf.reshape(combined_features, [combined_features.shape[0], -1])
        
        # Pass through dense layers with dropout, batch normalization, and regularization
        x = self.concat_dense1(combined_features)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.concat_dense2(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.concat_dense3(x)
        x = self.batch_norm3(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output
# Example usage
# model = fullimageModel()
# inputs = {
#     'text_embeddings': tf.random.normal([512, 12, 128]),
#     'color_histograms': tf.random.normal([512, 3, 256]),
#     'class_names_vectors': tf.random.uniform([512, 11], maxval=10, dtype=tf.int64),
#     'bboxes': tf.random.normal([512, 44]),
#     'confidences': tf.random.normal([512, 11])
# }
# output = model(inputs)
# print(output.shape)

# Example usage
# model = fullimageModel()
# inputs = {
#     'text_embeddings': tf.random.normal([512, 12, 128]),
#     'color_histograms': tf.random.normal([512, 3, 256]),
#     'class_names_vectors': tf.random.uniform([512, 11], maxval=10, dtype=tf.int64),
#     'bboxes': tf.random.normal([512, 44]),
#     'confidences': tf.random.normal([512, 11])
# }
# output = model(inputs)
# print(output.shape)



class fullimageModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        # Define the layers for the model
        self.class_name_embedding = tf.keras.layers.Embedding(input_dim=8, output_dim=128)  # Adjust input_dim as needed
        self.text_embedding_dense = tf.keras.layers.Dense(128, activation='elu')
        self.color_histogram_dense = tf.keras.layers.Dense(128, activation='elu')
        
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)  # Adjust num_heads and key_dim as needed
        self.concat_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.concat_dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.concat_dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='linear')  # Output layer for latitude and longitude

        # Regularization
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.l2_regularizer = tf.keras.regularizers.l2(0.01)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def call(self, inputs, mask=None):
        text_embeddings = inputs['text_embeddings']
        color_histograms = inputs['color_histograms']
        class_names = inputs['class_names_vectors']
        bboxes = inputs["bboxes"]
        
        # Truncate class_names to fixed size
        max_class_names = 11
        class_names = class_names[:, :max_class_names]

        # Convert class_names to one-hot encoding
        class_names_one_hot = tf.one_hot(class_names, depth=1000)
        
        # Create embeddings for class names
        class_name_embeddings = tf.reduce_sum(self.class_name_embedding(class_names_one_hot), axis=2)
        class_name_embeddings = tf.transpose(class_name_embeddings, perm=[0, 2, 1])
        
        # Apply dense layers to each feature
        text_embeddings = self.text_embedding_dense(text_embeddings)
        color_histograms = self.color_histogram_dense(color_histograms)
        
        # Combine text embeddings and class name embeddings for attention
        text_embeddings = tf.transpose(text_embeddings, perm=[0, 2, 1])  # Reshape to (batch_size, 128, 12)
        combined_embeddings = tf.concat([text_embeddings, class_name_embeddings], axis=2)
        
        # Apply multi-head attention mechanism
        attended_embeddings = self.multi_head_attention(query=combined_embeddings, value=combined_embeddings)
        attended_embeddings = tf.reshape(attended_embeddings, [tf.shape(attended_embeddings)[0], 128, -1])
        
        color_histograms = tf.reshape(color_histograms, [tf.shape(color_histograms)[0], 128, -1])
        
        # Concatenate all features along the 128 dimension
        combined_features = tf.concat([attended_embeddings, color_histograms], axis=2)
        
        # Flatten the combined features
        combined_features = tf.reshape(combined_features, [tf.shape(combined_features)[0], -1])
        
        # Pass through dense layers
        x = self.concat_dense1(combined_features)
        x = self.layer_norm1(x)
        x = self.concat_dense2(x)
        x = self.layer_norm2(x)
        x = self.concat_dense3(x)
        x = self.layer_norm3(x)
        output = self.output_layer(x)
        
        return output
