from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolo11n.pt')  # load a pretrained model

# Fine-tune the model with some transfer elarning yaddayadda
results = model.train(data='./datasets/large_signs_bb/data.yaml', epochs=100)  # train the model with your dataset
results = model.val()  # evaluate model performance on the validation set

# Save the trained model to a file
model.save('trained_model.pt')

print("Model training complete and saved as 'trained_model_100.pt'.")