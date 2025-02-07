import numpy as np
from transformers import BertTokenizer, BertModel
import torch
'''
I dont like the names and I dont like the dir but pytohn wouldn't find the file anywhere else ? 
 __ init __ . py issue ti think
'''
class embedder:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.embeddings = {}

    def fit(self, class_freq_dict):
        for class_name, freq in class_freq_dict.items():
            self.embeddings[class_name] = np.random.rand(self.embedding_size) * freq

    def transform(self, class_name):
        return self.embeddings.get(class_name, np.zeros(self.embedding_size))

# Example usage:
# embedder = OBJEmbedder(embedding_size=10)
# class_freq_dict = {'class1': 5, 'class2': 3, 'class3': 8}
# embedder.fit(class_freq_dict)
# print(embedder.transform('class1'))

class TextEmbedder(embedder):
    def __init__(self, embedding_size, bert_model_name='bert-base-uncased'):
        super().__init__(embedding_size)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

# Example usage:
# text_embedder = TextEmbedder(embedding_size=10)
# text_embedding = text_embedder.embed_text("This is a sample text.")
# print(text_embedding)