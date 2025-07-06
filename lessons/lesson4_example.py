import torch
from lessons.lesson3_datasets import get_mnist_dataloaders
from lessons.lesson4_models import SimpleCNN, CNNWithResidual
from lessons.lesson3_trainer import train_model
from lessons.lesson4_utils import plot_training_history, count_parameters, compare_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_loader, test_loader = get_mnist_dataloaders(batch_size=64)

simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
residual_cnn = CNNWithResidual(input_channels=1, num_classes=10).to(device)

print(f'Simple CNN params: {count_parameters(simple_cnn)}')
print(f'Residual CNN params: {count_parameters(residual_cnn)}')

simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=5, device=str(device))
residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=5, device=str(device))

compare_models(simple_history, residual_history)