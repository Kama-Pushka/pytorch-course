import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from lessons.lesson3_datasets import get_cifar_dataloaders
from lessons.lesson3_trainer import train_model, count_parameters
from lessons.lesson4_models import SimpleCNN, CIFARCNN

### 3.1 Реализация кастомных слоев (15 баллов)
# Реализуйте кастомные слои:
# - Кастомный сверточный слой с дополнительной логикой
# - Attention механизм для CNN
# - Кастомная функция активации
# - Кастомный pooling слой
#
# Для каждого слоя:
# - Реализуйте forward и backward проходы
# - Добавьте параметры если необходимо
# - Протестируйте на простых примерах
# - Сравните с стандартными аналогами

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, noise_stddev=0.1):
        super(CustomConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.noise_stddev = noise_stddev

    def forward(self, x):
        # Сворачиваем обычное изображение
        x = self.conv(x)
        # Добавляем гауссов шум
        noise = torch.randn_like(x) * self.noise_stddev
        noisy_output = x + noise
        return noisy_output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Получаем среднюю и максимальную активации по каналам
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale_map = torch.cat([avg_out, max_out], dim=1)
        attn_mask = self.conv(scale_map)
        attn_mask = self.sigmoid(attn_mask)
        return x * attn_mask

# Parametric Exponential Linear Unit
class PELU(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(PELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        positive_part = torch.where(x >= 0, x, torch.zeros_like(x))
        negative_part = torch.where(x < 0, self.alpha * (torch.exp(x / self.beta) - 1), torch.zeros_like(x))
        return positive_part + negative_part

class HybridPooling(nn.Module):
    def __init__(self):
        super(HybridPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(2, 2)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        avg_pooled = self.avgpool(x)
        max_pooled = self.maxpool(x)
        hybrid_pooled = 0.5 * (avg_pooled + max_pooled)
        return hybrid_pooled

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = CustomConv2d(3, 32, kernel_size=3, padding=1)
        self.attn1 = SpatialAttention()
        self.act1 = PELU()
        self.pool1 = HybridPooling()
        self.conv2 = CustomConv2d(32, 64, kernel_size=3, padding=1)
        self.attn2 = SpatialAttention()
        self.act2 = PELU()
        self.pool2 = HybridPooling()
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "1__main__":
    # Проверка кастомного свёрточного слоя
    custom_conv = CustomConv2d(3, 64, kernel_size=3, padding=1)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = custom_conv(input_tensor)
    print("Custom Conv:", output.shape)

    # Проверка механизма Attention
    spatial_attn = SpatialAttention()
    output_attention = spatial_attn(input_tensor)
    print("Spatial Attention:", output_attention.shape)

    # Проверка кастомной функции активации
    pelu_activation = PELU()
    activated_tensor = pelu_activation(input_tensor)
    print("PELU Activated Tensor:", activated_tensor.shape)

    # Проверка гибридного пулинга
    hybrid_pool = HybridPooling()
    pooled_tensor = hybrid_pool(input_tensor)
    print("Hybrid Pooling:", pooled_tensor.shape)

    # Получаем загрузчики данных
    train_loader, test_loader = get_cifar_dataloaders(batch_size=64)

    # Тестируем обе модели
    custom_model = CustomCNN()
    standard_model = CIFARCNN()

    print("Training Custom CNN...")
    train_model(custom_model, train_loader, test_loader)

    print("\nTraining Standard CNN...")
    train_model(standard_model, train_loader, test_loader)

### 3.2 Эксперименты с Residual блоками (15 баллов)
# Исследуйте различные варианты Residual блоков:
# - Базовый Residual блок
# - Bottleneck Residual блок
# - Wide Residual блок
#
# Для каждого варианта:
# - Реализуйте блок с нуля
# - Сравните производительность
# - Проанализируйте количество параметров
# - Исследуйте стабильность обучения

class BasicResidualBlock(nn.Module):
    expansion = 1  # Коэффициент расширения (число каналов)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class BottleneckResidualBlock(nn.Module):
    expansion = 4  # Увеличение числа каналов в последнем слое

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class WideResidualBlock(nn.Module):
    expansion = 1  # Не расширяем каналы

    def __init__(self, inplanes, planes, stride=1, downsample=None, widening_factor=2):
        super(WideResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*widening_factor, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*widening_factor)
        self.conv2 = nn.Conv2d(planes*widening_factor, planes*widening_factor, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*widening_factor)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class BaselineResNet(nn.Module):
    def __init__(self, block_type=BasicResidualBlock, num_classes=10):
        super(BaselineResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Прямая инициализация слоев
        self.layer1 = nn.Sequential(
            block_type(64, 64, stride=1),
            block_type(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            block_type(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128)
            )),
            block_type(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            block_type(128, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )),
            block_type(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            block_type(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512)
            )),
            block_type(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BottleneckResNet(nn.Module):
    def __init__(self, block_type=BottleneckResidualBlock, num_classes=10):
        super(BottleneckResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Первым делом настраиваем первый блок с downsampling
        self.layer1 = nn.Sequential(
            block_type(64, 64, stride=1, downsample=nn.Sequential(
                nn.Conv2d(64, 64 * block_type.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64 * block_type.expansion)
            )),  # Обратите внимание на расширение каналов
            block_type(64 * block_type.expansion, 64, stride=1)  # Оставшиеся блоки совпадают по размерам
        )

        # Другие слои остаются теми же
        self.layer2 = nn.Sequential(
            block_type(64 * block_type.expansion, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64 * block_type.expansion, 128 * block_type.expansion, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128 * block_type.expansion)
            )),
            block_type(128 * block_type.expansion, 128, stride=1)
        )

        self.layer3 = nn.Sequential(
            block_type(128 * block_type.expansion, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(128 * block_type.expansion, 256 * block_type.expansion, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256 * block_type.expansion)
            )),
            block_type(256 * block_type.expansion, 256, stride=1)
        )

        self.layer4 = nn.Sequential(
            block_type(256 * block_type.expansion, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256 * block_type.expansion, 512 * block_type.expansion, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512 * block_type.expansion)
            )),
            block_type(512 * block_type.expansion, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WideResNet(nn.Module):
    def __init__(self, block_type=WideResidualBlock, num_classes=10, widening_factor=2):
        super(WideResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Инициализируем слои с downsample для корректировки каналов
        self.layer1 = nn.Sequential(
            block_type(64, 64, stride=1, widening_factor=widening_factor, downsample=nn.Sequential(
                nn.Conv2d(64, 64 * widening_factor, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(64 * widening_factor)
            )),
            block_type(64 * widening_factor, 64, stride=1, widening_factor=widening_factor)
        )

        self.layer2 = nn.Sequential(
            block_type(64 * widening_factor, 128, stride=2, widening_factor=widening_factor, downsample=nn.Sequential(
                nn.Conv2d(64 * widening_factor, 128 * widening_factor, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128 * widening_factor)
            )),
            block_type(128 * widening_factor, 128, stride=1, widening_factor=widening_factor)
        )

        self.layer3 = nn.Sequential(
            block_type(128 * widening_factor, 256, stride=2, widening_factor=widening_factor, downsample=nn.Sequential(
                nn.Conv2d(128 * widening_factor, 256 * widening_factor, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256 * widening_factor)
            )),
            block_type(256 * widening_factor, 256, stride=1, widening_factor=widening_factor)
        )

        self.layer4 = nn.Sequential(
            block_type(256 * widening_factor, 512, stride=2, widening_factor=widening_factor, downsample=nn.Sequential(
                nn.Conv2d(256 * widening_factor, 512 * widening_factor, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512 * widening_factor)
            )),
            block_type(512 * widening_factor, 512, stride=1, widening_factor=widening_factor)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * widening_factor, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Получаем загрузчики данных
    train_loader, test_loader = get_cifar_dataloaders(batch_size=64)

    # Training the models
    base_model = BaselineResNet()
    bottle_model = BottleneckResNet()
    wide_model = WideResNet(widening_factor=2)

    # Выводы
    print("Параметры:")
    print(f'- BaselineResNet: {count_parameters(base_model):,}')
    print(f'- BottleneckResNet: {count_parameters(bottle_model):,}')
    print(f'- WideResNet: {count_parameters(wide_model):,}')

    print("Training Baseline ResNet...")
    start_time = time.time()
    train_model(base_model, train_loader, test_loader)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")

    print("\nTraining Bottleneck ResNet...")
    start_time = time.time()
    train_model(bottle_model, train_loader, test_loader)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")

    print("\nTraining Wide ResNet...")
    start_time = time.time()
    train_model(wide_model, train_loader, test_loader)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} s")