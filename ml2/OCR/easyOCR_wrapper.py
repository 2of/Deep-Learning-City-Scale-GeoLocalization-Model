import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Read text from an image
results = reader.readtext('sample_data/samplestreetviews/turkey.jpeg')

# Extract the detected text
detected_texts = [text for (_, text, _) in results]

print(detected_texts)


class EasyOCRWrapper():
    def __init__(self):
        self.reader = reader = easyocr.Reader(['en'], gpu=True)
    
    
    def predict(self,image):
        results = reader.readtext(image)
        detected_texts = [text for (_, text, _) in results]

        return detected_texts
    
        