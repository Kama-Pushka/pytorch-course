import kagglehub

# Download latest version
path = kagglehub.dataset_download("rahimanshu/cardiomegaly-disease-prediction-using-cnn")

print("Path to dataset files:", path)