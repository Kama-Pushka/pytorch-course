import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class CardiomegalyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b4')
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.base_model._fc.in_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.base_model._fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)  # Логарифмическая Softmax для кросс-энтропии
        )

    def forward(self, x):
        # Через EfficientNet получаем признаки
        features = self.base_model.extract_features(x)
        # Применяем глобальный max pooling
        pooled_features = nn.functional.adaptive_max_pool2d(features, 1)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        # Прогоняем через головной слой
        output = self.head(flattened_features)
        return output

# class CardiomegalyClassifier(nn.Module):
#     def __init__(self):
#         super(CardiomegalyClassifier, self).__init__()
#         # Входное изображение имеет форму (N, C, H, W), где C=3 (цветное изображение)
#         # Первые слои свертывания и Pooling
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#
#         # Полносвязные слои
#         self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Выход после Pooling слоя будет уменьшен до 28x28
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(512, 1)  # Бинарная классификация (одно выходное значение)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#
#         # Переупаковка (flatten) признаков перед подачей в полносвязные слои
#         x = x.view(-1, 128 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return torch.sigmoid(x)  # Возвращаем вероятность принадлежности к классу "патологии"