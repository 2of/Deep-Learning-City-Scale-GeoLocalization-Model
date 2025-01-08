import easyocr
from sentence_transformers import SentenceTransformer
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
class EasyOCRWrapper():
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def predict(self, image):
        # print(type(image))
        results = self.reader.readtext(image)
        detected_texts = [text for (_, text, _) in results]
        return detected_texts

    def create_text_embedding(self, texts, embedding_size=128, max_predictions=12):
        # Convert list of texts to embeddings
        embeddings = self.model.encode(texts)
        # Ensure all embeddings are the same size of 12x384
        fixed_size_embeddings = np.zeros((max_predictions, embedding_size))
        for i, embedding in enumerate(embeddings[:max_predictions]):
            if len(embedding) > embedding_size:
                fixed_size_embeddings[i] = embedding[:embedding_size]
            else:
                fixed_size_embeddings[i, :len(embedding)] = embedding
        return torch.tensor(fixed_size_embeddings)

    def predict_and_embed(self, image, embedding_size=384, max_predictions=12):
        detected_texts = self.predict(image)
        text_embeddings = self.create_text_embedding(detected_texts, embedding_size, max_predictions)
        return detected_texts, text_embeddings

    def predict_and_embed(self, image):
        detected_texts = self.predict(image)
        text_embeddings = self.create_text_embedding(detected_texts)
        return detected_texts, text_embeddings

        
 
    def predict_and_embed_from_group_as_tensor(self, tensor_of_images): 
        # descriptive, ik
        print("_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+")
        print(type(tensor_of_images))
        print(tensor_of_images.shape)
        
        all_detected_texts = []
        all_text_embeddings = []
        
        # Iterate over each image in the tensor
        for i in range(tensor_of_images.shape[0]):
            # Separate out the current image
            current_image = tensor_of_images[i, :, :, :]
            print(f"Image {i} shape:", current_image.shape)
            
            # Convert the tensor to a numpy array and transpose it to (H, W, C) format for displaying
            current_image_np = current_image.permute(1, 2, 0).numpy()
            
            # Ensure the image is in the correct format for EasyOCR
            current_image_np = (current_image_np * 255).astype(np.uint8)
            
            # # Display the image using matplotlib
            # plt.imshow(current_image_np)
            # plt.axis('off')  # Hide the axis
            # plt.show()
            
            # Process the current image
            detected_texts, text_embeddings = self.predict_and_embed(current_image_np)
            print(f"GOT for image {i}: ", detected_texts)
            
            # Append the detected texts and embeddings to the lists
            all_detected_texts.append(detected_texts)
            all_text_embeddings.append(text_embeddings)
        
        # Stack all text embeddings into a single tensor
        stacked_text_embeddings = torch.stack(all_text_embeddings)
        
        return all_detected_texts, stacked_text_embeddings

               
    
# Example usage:
ocr = EasyOCRWrapper()
detected_texts, text_embeddings = ocr.predict_and_embed('sample_data/samplestreetviews/turkey.jpeg')
print(detected_texts)
print(text_embeddings.shape)


detected_texts, text_embeddings = ocr.predict_and_embed('sample_data/samplesign.png')

print(detected_texts)
print(text_embeddings.shape)

