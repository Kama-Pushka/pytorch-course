import torch
from torchinfo import summary

from datasets import get_cardiomegaly_dataloaders
from models import CardiomegalyClassifier
from trainer import train_model
from utils import visualize_first_batch, plot_training_validation_curves, count_parameters, compute_confusion_matrix

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_path = 'data/cardiomegaly-disease-prediction-using-cnn/train/train'
    test_data_path = 'data/cardiomegaly-disease-prediction-using-cnn/test/test'

    model = CardiomegalyClassifier()
    train_loader, test_loader = get_cardiomegaly_dataloaders(train_data_path, test_data_path, recursive=False)

    for inputs, labels in train_loader:
        print(labels[:5])  # Первые пять меток
        print(inputs.shape, labels.shape)  # Должно быть (batch_size, channels, width, height) и (batch_size,)
        summary(model, input_size=inputs.shape)
        break
    visualize_first_batch(train_loader)

    metrics = train_model(model, train_loader, test_loader, epochs=10, lr=0.0005)

    plot_training_validation_curves(metrics)
    compute_confusion_matrix(model, test_loader, device)