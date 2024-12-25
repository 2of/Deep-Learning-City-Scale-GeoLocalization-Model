import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Read text from an image
results = reader.readtext('sample_data/samplestreetviews/turkey.jpeg')

# Extract the detected text
detected_texts = [text for (_, text, _) in results]

print(detected_texts)


class EasyOCRWrapper():
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
    
    
    def predict(self,image):
        results = reader.readtext(image)
        detected_texts = [text for (_, text, _) in results]

        return detected_texts
    
        