import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Read text from an image
results = reader.readtext('res/samplestreetviews/turkey.jpeg')

# Extract the detected text
detected_texts = [text for (_, text, _) in results]

print(detected_texts)