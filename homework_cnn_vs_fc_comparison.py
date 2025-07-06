import numpy as np
import torch
from matplotlib import pyplot as plt
import time

from lessons.lesson3_datasets import get_mnist_dataloaders, get_cifar_dataloaders
from lessons.lesson3_models import FCN
from lessons.lesson4_models import SimpleCNN, CNNWithResidual
from lessons.lesson3_trainer import train_model
from lessons.lesson4_utils import plot_training_history, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# TODO скорее всего не совсем корректное измерение времени

### 1.1 Сравнение на MNIST (20 баллов)
# Сравните производительность на MNIST:
# - Полносвязная сеть (3-4 слоя)
# - Простая CNN (2-3 conv слоя)
# - CNN с Residual Block
#
# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность на train и test множествах
# - Измерьте время обучения и инференса
# - Визуализируйте кривые обучения
# - Проанализируйте количество параметров

def measure_inference_time(model, loader, device):
    """Измеряет среднее время инференса одной партии."""
    model.eval()
    times = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            start_time = time.time()
            model(images)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
    return sum(times) / len(times)

def compare_models(fc_history, simple_cnn_history, cnn_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Кривые ошибок
    axes[0].plot(fc_history['test_losses'], label='FC Network', marker='o')
    axes[0].plot(simple_cnn_history['test_losses'], label='CNN', marker='s')
    axes[0].plot(cnn_history['test_losses'], label='CNN with Residual Block', marker='^')
    axes[0].set_title('Test Loss Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # Кривые точности
    axes[1].plot(fc_history['test_accs'], label='FC Network', marker='o')
    axes[1].plot(simple_cnn_history['test_accs'], label='CNN', marker='s')
    axes[1].plot(cnn_history['test_accs'], label='CNN with Residual Block', marker='^')
    axes[1].set_title('Test Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "1__main__":
    # Конфигурация полносвязной модели
    config = {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'}
        ]
    }

    # Получаем загрузчики данных
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)

    # Полносвязная модель (3 слоя)
    fc_model = FCN(**config).to(device)
    start_fc = time.time()
    fc_history = train_model(fc_model, train_loader, test_loader, epochs=10, lr=0.001, device=str(device))
    end_fc = time.time()
    fc_inference_time = measure_inference_time(fc_model, test_loader, device)

    # Простая CNN (2 conv слоя)
    simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
    start_simple = time.time()
    simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=10, lr=0.001, device=str(device))
    end_simple = time.time()
    simple_inference_time = measure_inference_time(simple_cnn, test_loader, device)

    # CNN с Residual Block
    residual_cnn = CNNWithResidual(input_channels=1, num_classes=10).to(device)
    start_residual = time.time()
    residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=10, lr=0.001, device=str(device))
    end_residual = time.time()
    residual_inference_time = measure_inference_time(residual_cnn, test_loader, device)

    # Анализируем количество параметров
    fc_params = count_parameters(fc_model)
    simple_cnn_params = count_parameters(simple_cnn)
    residual_cnn_params = count_parameters(residual_cnn)

    # Время обучения
    training_times = {
        'FCN': end_fc - start_fc,
        'Simple CNN': end_simple - start_simple,
        'CNN with Residual': end_residual - start_residual
    }

    # Выводы
    print("Параметры:")
    print(f'- FCN: {fc_params:,}')
    print(f'- Simple CNN: {simple_cnn_params:,}')
    print(f'- CNN with Residual: {residual_cnn_params:,}')

    print("\nВремя обучения:")
    for k, v in training_times.items():
        print(f"- {k}: {v:.2f} сек.")

    print("\nСреднее время инференса на одну партию:")
    print(f'- FCN: {fc_inference_time:.4f} сек.')
    print(f'- Simple CNN: {simple_inference_time:.4f} сек.')
    print(f'- CNN with Residual: {residual_inference_time:.4f} сек.')

    # Сравниваем модели графически
    compare_models(fc_history, simple_history, residual_history)

### 1.2 Сравнение на CIFAR-10 (20 баллов)
# Сравните производительность на CIFAR-10:
# - Полносвязная сеть (глубокая)
# - CNN с Residual блоками
# - CNN с регуляризацией и Residual блоками
#
# Для каждого варианта:
# - Обучите модель с одинаковыми гиперпараметрами
# - Сравните точность и время обучения
# - Проанализируйте переобучение
# - Визуализируйте confusion matrix
# - Исследуйте градиенты (gradient flow)

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(CNNWithResidual, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.linear = nn.Linear(256*4, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class CNNWithRegularizationAndResidual(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(CNNWithRegularizationAndResidual, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.2)  # Regularization via dropout
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.dropout2 = nn.Dropout2d(0.2)  # Additional regularization layer
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.dropout3 = nn.Dropout2d(0.2)  # More dropout to prevent overfitting
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.linear = nn.Linear(256*4, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.layer1(out)
        out = self.dropout2(out)
        out = self.layer2(out)
        out = self.dropout3(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Analyze confusion matrices
def compute_confusion_matrix(model, loader):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(conf_mat_fcn, conf_mat_cnn_with_residual, conf_mat_cnn_with_reg_and_residual):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(conf_mat_fcn, annot=True, fmt='d', cbar=False, ax=axes[0])
    axes[0].set_title('Deep Fully Connected')
    sns.heatmap(conf_mat_cnn_with_residual, annot=True, fmt='d', cbar=False, ax=axes[1])
    axes[1].set_title('CNN with Residual')
    sns.heatmap(conf_mat_cnn_with_reg_and_residual, annot=True, fmt='d', cbar=False, ax=axes[2])
    axes[2].set_title('CNN with Reg & Residual')
    plt.show()

# Visualizing gradient flows
def visualize_gradient_flow(model):
    grad_mags = []
    for param in model.named_parameters():
        if param[1].grad is not None:
            grad_mag = param[1].grad.norm().item()
            grad_mags.append(grad_mag)
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(grad_mags)), grad_mags)
    plt.title('Gradient Flow Analysis')
    plt.xlabel('Layer index')
    plt.ylabel('Gradient magnitude')
    plt.show()

if __name__ == "__main__":
    # Конфигурация полносвязной модели
    config = {
        'input_size': 3072,  # Размерность ввода (32x32x3)
        'num_classes': 10,  # Число классов (10 категорий)
        'layers': [
            {'type': 'linear', 'size': 2048},  # Первый слой
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.5},  # Регуляризация Dropout
            {'type': 'linear', 'size': 1024},  # Второй слой
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.5},  # Регуляризация Dropout
            {'type': 'linear', 'size': 512},  # Третий слой
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.5},  # Регуляризация Dropout
            {'type': 'linear', 'size': 256},  # Четвертый слой
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.5},  # Регуляризация Dropout
            {'type': 'linear', 'size': 128},  # Пятый слой
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.5},  # Регуляризация Dropout
            {'type': 'linear', 'size': 64},  # Шестой слой
            {'type': 'relu'},
            {'type': 'dropout', 'rate': 0.5},  # Регуляризация Dropout
            {'type': 'linear', 'size': 10},  # Выходной слой
        ],
    }

    # Получаем загрузчики данных
    train_loader, test_loader = get_cifar_dataloaders(batch_size=64)

    # Полносвязная модель (Глубокая)
    fc_model = FCN(**config).to(device)
    start_fc = time.time() # TODO !!
    fc_history = train_model(fc_model, train_loader, test_loader, epochs=10, lr=0001., device=str(device))
    end_fc = time.time()

    # Простая CNN (2 conv слоя)
    cnn_with_residual = CNNWithResidual(input_channels=3, num_classes=10).to(device)
    start_simple = time.time()
    simple_history = train_model(cnn_with_residual, train_loader, test_loader, epochs=10, lr=0.001, device=str(device))
    end_simple = time.time()

    # CNN с Residual Block
    cnn_with_reg_and_residual = CNNWithRegularizationAndResidual(input_channels=3, num_classes=10).to(device)
    start_residual = time.time()
    residual_history = train_model(cnn_with_reg_and_residual, train_loader, test_loader, epochs=10, lr=0.001, device=str(device))
    end_residual = time.time()

    # Сравниваем модели графически
    compare_models(fc_history, simple_history, residual_history)

    conf_mat_fcn = compute_confusion_matrix(fc_model, test_loader)
    conf_mat_cnn_with_residual = compute_confusion_matrix(cnn_with_residual, test_loader)
    conf_mat_cnn_with_reg_and_residual = compute_confusion_matrix(cnn_with_reg_and_residual, test_loader)
    plot_confusion_matrix(conf_mat_fcn, conf_mat_cnn_with_residual, conf_mat_cnn_with_reg_and_residual)

    visualize_gradient_flow(fc_model)
    visualize_gradient_flow(cnn_with_residual)
    visualize_gradient_flow(cnn_with_reg_and_residual)

    ###

    # Анализируем количество параметров
    fc_params = count_parameters(fc_model)
    simple_cnn_params = count_parameters(cnn_with_residual)
    residual_cnn_params = count_parameters(cnn_with_reg_and_residual)

    # Время обучения
    training_times = {
        'FCN': end_fc - start_fc,
        'Simple CNN': end_simple - start_simple,
        'CNN with Residual': end_residual - start_residual
    }

    # Выводы
    print("Параметры:")
    print(f'- FCN: {fc_params:,}')
    print(f'- Simple CNN: {simple_cnn_params:,}')
    print(f'- CNN with Residual: {residual_cnn_params:,}')

    print("\nВремя обучения:")
    for k, v in training_times.items():
        print(f"- {k}: {v:.2f} сек.")