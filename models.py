import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class CardiomegalyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b3') # num_classes=2
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.base_model._fc.in_features),
            nn.Linear(self.base_model._fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)  # Логарифмическая Softmax для кросс-энтропии
        )

    def forward(self, x):
        features = self.base_model.extract_features(x) # Через EfficientNet получаем признаки
        pooled_features = nn.functional.adaptive_max_pool2d(features, 1) # Применяем глобальный max pooling
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.head(flattened_features) # Прогоняем через головной слой
        return output