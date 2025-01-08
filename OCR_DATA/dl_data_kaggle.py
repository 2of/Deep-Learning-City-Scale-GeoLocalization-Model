import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/road-sign-detection")

print("Path to dataset files:", path)