import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tabulate import tabulate

from lessons.lesson3_datasets import get_cifar_dataloaders
from lessons.lesson3_trainer import train_model
from lessons.lesson4_models import CIFARCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

### 2.1 Влияние размера ядра свертки (15 баллов)
# Исследуйте влияние размера ядра свертки:
# - 3x3 ядра
# - 5x5 ядра
# - 7x7 ядра
# - Комбинация разных размеров (1x1 + 3x3)
#
# Для каждого варианта:
# - Поддерживайте одинаковое количество параметров
# - Сравните точность и время обучения
# - Проанализируйте рецептивные поля
# - Визуализируйте активации первого слоя

class CIFARCNNKernel5x5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)  # 5x5 kernel, same padding
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFARCNNKernel7x7(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)  # 7x7 kernel, same padding
        self.conv2 = nn.Conv2d(32, 64, 7, 1, 3)
        self.conv3 = nn.Conv2d(64, 128, 7, 1, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFARCNNCombined(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1a = nn.Conv2d(3, 32, 1, 1, 0)  # 1x1 kernel
        self.conv1b = nn.Conv2d(32, 32, 3, 1, 1)  # 3x3 kernel
        self.conv2a = nn.Conv2d(32, 64, 1, 1, 0)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3a = nn.Conv2d(64, 128, 1, 1, 0)
        self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1b(F.relu(self.conv1a(x)))))  # Combined convolutions
        x = self.pool(F.relu(self.conv2b(F.relu(self.conv2a(x)))))
        x = self.pool(F.relu(self.conv3b(F.relu(self.conv3a(x)))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def calculate_receptive_field(conv_layers):
    """Подсчет количества рецептивных полей"""
    rf = 1  # Initial receptive field
    stride = 1
    for layer in conv_layers:
        kernel_size = layer.kernel_size[0]
        dilation = layer.dilation[0]
        rf = rf + ((kernel_size - 1) * dilation)
        stride *= layer.stride[0]
    return rf

def show_first_layer_activations(model, sample_images):
    """Визуализация активации первого слоя"""
    first_conv_layer = list(model.modules())[1]  # Assuming it's the first Conv2D layer
    act_fn = lambda x: F.relu(first_conv_layer(x)).detach().cpu().numpy()[0][0]  # Take first feature map
    activations = [act_fn(img.unsqueeze(0)) for img in sample_images]
    plt.figure(figsize=(10, 4))
    for idx, act in enumerate(activations):
        plt.subplot(1, len(sample_images), idx+1)
        plt.imshow(act, cmap='viridis')
        plt.axis('off')
    plt.show()

if __name__ == "1__main__":
    train_loader, test_loader = get_cifar_dataloaders(batch_size=64)

    models = {
        '3x3 Kernel': CIFARCNN(num_classes=10),
        '5x5 Kernel': CIFARCNNKernel5x5(num_classes=10),
        '7x7 Kernel': CIFARCNNKernel7x7(num_classes=10),
        'Mixed Kernel (1x1 + 3x3)': CIFARCNNCombined(num_classes=10)
    }

    sample_images = next(iter(test_loader))[0][:4].to(device)  # 4 изображения для визуализации активации

    results = {}
    for name, model in models.items():
        print(f'Training {name}...')
        start_time = time.time()
        results[name] = train_model(model, train_loader, test_loader)
        end_time = time.time()
        results[name]['time'] = end_time - start_time
        print(f'{name} trained in {results[name]['time']:.2f} seconds')
        print(f'Final Test Accuracy: {results[name]['test_accs']}%\n')
        show_first_layer_activations(model, sample_images)

    # Таблица результатов обучения
    table_data = [[name, round(results[name]['time'], 2), results[name]['test_accs']] for name in models.keys()]
    headers = ['Model Name', 'Training Time (sec)', 'Final Test Accuracy']
    print('\nSummary Table:')
    print(tabulate(table_data, headers=headers, tablefmt='orgtbl'))

    # Вычисление рецептивных полей
    receptive_fields = {}
    for name, model in models.items():
        conv_layers = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
        receptive_fields[name] = calculate_receptive_field(conv_layers)

    print('\nReceptive Fields:')
    for name, rf in receptive_fields.items():
        print(f"{name}: {rf}x{rf}")

    # for name, result in results.items():
    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(result['train_losses'], label=name + ' Train Loss')
    #     plt.title('Training Loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #
    #     plt.subplot(1, 2, 2)
    #     plt.plot(result['train_accs'], label=name + ' Train Accuracy')
    #     plt.title('Training Accuracy')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
    #
    #     plt.show()

### 2.2 Влияние глубины CNN (15 баллов)
# Исследуйте влияние глубины CNN:
# - Неглубокая CNN (2 conv слоя)
# - Средняя CNN (4 conv слоя)
# - Глубокая CNN (6+ conv слоев)
# - CNN с Residual связями
#
# Для каждого варианта:
# - Сравните точность и время обучения
# - Проанализируйте vanishing/exploding gradients
# - Исследуйте эффективность Residual связей
# - Визуализируйте feature maps

class ShallowCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MediumCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*2*2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Используем adaptive pooling
        self.fc1 = nn.Linear(512*4*4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 64, 2, 1)
        self.layer2 = self.make_layer(64, 128, 2, 2)
        self.layer3 = self.make_layer(128, 256, 2, 2)
        self.layer4 = self.make_layer(256, 512, 2, 2)
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def check_gradients(model):
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.norm().item())
    return gradients

def visualize_feature_maps(model, input_image):
    """
    Метод для визуализации активаций всех свёрточных слоёв для одного изображения.
    :param model: Объект модели CNN
    :param input_image: Одно изображение из датасета
    """
    activations = []
    handle_inputs = input_image.unsqueeze(0).to(device)  # Добавляем первую размерность (batch-size)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # Берём копию входных данных, чтобы не испортить оригинал
            temp_inputs = handle_inputs.clone()
            try:
                handle_inputs = module(temp_inputs)
                activations.append(handle_inputs.detach().cpu().numpy()[0])  # Сохраняем активацию
            except Exception as e:
                print(f"Error applying module: {e}")
                break

    num_layers = len(activations)
    rows = int(np.ceil(np.sqrt(num_layers)))  # Распределяем равномерно по строкам и столбцам
    cols = int(np.ceil(num_layers / rows))

    plt.figure(figsize=(cols * 4, rows * 4))
    for i, act in enumerate(activations):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(act.sum(axis=0), cmap='viridis')  # Суммируем каналы, чтобы получить одно изображение
        plt.title(f'Conv Layer {i+1}', fontsize=12)
        plt.axis('off')
    plt.suptitle('Activation Maps per Convolutional Layer', fontsize=16)
    plt.show()

if __name__ == "__main__":
    train_loader, test_loader = get_cifar_dataloaders(batch_size=64)

    models = {
        'Shallow CNN (2 conv layers)': ShallowCNN(num_classes=10),
        'Medium CNN (4 conv layers)': MediumCNN(num_classes=10),
        'Deep CNN (6 conv layers)': DeepCNN(num_classes=10),
        'ResNet-like CNN': ResNetCNN(num_classes=10)
    }

    results = {}
    for name, model in models.items():
        print(f'\nTraining {name}...')
        start_time = time.time()
        results[name] = train_model(model, train_loader, test_loader)
        end_time = time.time()
        results[name]['time'] = end_time - start_time

    # Таблица результатов
    final_results = [['Model', 'Training Time (sec)', 'Final Test Accuracy']]
    for name, res in results.items():
        final_results.append([name, round(res['time'], 2), res['test_accs']])

    print('\nSummary Table:')
    print(
        tabulate(final_results, headers=['Model', 'Training Time (seconds)', 'Final Test Accuracy'], tablefmt='pretty'))

    # График градиентов
    gradients = {}
    for name, model in models.items():
        gradients[name] = check_gradients(model)

    plt.figure(figsize=(10, 5))
    for name, g_list in gradients.items():
        plt.hist(g_list, bins=50, alpha=0.5, label=name)
    plt.title('Histogram of Gradient Norms')
    plt.xlabel('Gradient Norm')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Feature maps
    random_sample = next(iter(test_loader))[0][0].to(device)
    for name, model in models.items():
        print(f'\nVisualizing feature maps for {name}:')
        visualize_feature_maps(model, random_sample)