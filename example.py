import torch
from torchinfo import summary

from datasets import get_cardiomegaly_dataloaders
from models import CardiomegalyClassifier
from trainer import train_model
from utils import visualize_first_batch, plot_training_validation_curves, count_parameters, compute_confusion_matrix

# sns.set_style('whitegrid')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO а надо ли?

    train_data_path = 'data/cardiomegaly-disease-prediction-using-cnn/train/train'
    test_data_path = 'data/cardiomegaly-disease-prediction-using-cnn/test/test'

    # Инициализация модели и DataLoaders
    model = CardiomegalyClassifier()
    train_loader, test_loader = get_cardiomegaly_dataloaders(train_data_path, test_data_path)

    for inputs, labels in train_loader:
        print(labels[:5])  # Посмотрите на первые пять меток
        print(inputs.shape, labels.shape)  # Должно быть (batch_size, channels, width, height) и (batch_size,)
        summary(model, input_size=inputs.shape)
        break

    visualize_first_batch(train_loader)

    # Обучение и проверка модели
    metrics = train_model(model, train_loader, test_loader, epochs=10, lr=0.0005)

    # Визуализация кривых обучения и валидации
    plot_training_validation_curves(metrics)

    compute_confusion_matrix(model, test_loader, device)