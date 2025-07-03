import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from utils.lesson3_datasets import get_mnist_dataloaders
from utils.lesson3_models import FCN
from utils.lesson3_trainer import train_model
from lessons.utils import plot_training_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 3.1 Сравнение техник регуляризации (15 баллов)
# Исследуйте различные техники регуляризации:
# - Без регуляризации
# - Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
# - Только BatchNorm
# - Dropout + BatchNorm
# - L2 регуляризация (weight decay)
#
# Для каждого варианта:
# - Используйте одинаковую архитектуру
# - Сравните финальную точность
# - Проанализируйте стабильность обучения
# - Визуализируйте распределение весов

# Общая архитектура модели
base_model_config = [
    {"type": "linear", "size": 512},
    {"type": "relu"},
    {"type": "linear", "size": 256},
    {"type": "relu"},
    {"type": "linear", "size": 128},
    {"type": "relu"},
    {"type": "linear", "size": 10}
]

regularization_techniques = [
    {"name": "Без регуляризации", "config": base_model_config.copy()},
    {"name": "Dropout (p=0.1)",
     "config": base_model_config[:3] + [{"type": "dropout", "rate": 0.1}] + base_model_config[3:]},
    {"name": "Dropout (p=0.3)",
     "config": base_model_config[:3] + [{"type": "dropout", "rate": 0.3}] + base_model_config[3:]},
    {"name": "Dropout (p=0.5)",
     "config": base_model_config[:3] + [{"type": "dropout", "rate": 0.5}] + base_model_config[3:]},
    {"name": "BatchNorm", "config": base_model_config[:3] + [{"type": "batch_norm"}] + base_model_config[3:]},
    {"name": "Dropout+BatchNorm",
     "config": base_model_config[:3] + [{"type": "dropout", "rate": 0.3}, {"type": "batch_norm"}] + base_model_config[3:]},
    {"name": "L2 Регуляризация", "config": base_model_config.copy()}
]

# Функция оценки веса модели
def analyze_weights(model):
    weights = []
    for p in model.parameters():
        if p.dim() > 1:  # Берём только веса, игнорируя смещения
            weights.extend(p.detach().cpu().numpy().flatten())
    return weights

def regularization_test():
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)

    results = []
    for technique in regularization_techniques:
        print(f"Эксперимент с техникой: {technique['name']}")

        model = FCN(input_size=784, num_classes=10, layers=technique["config"]).to(device)

        optimizer = optim.Adam(model.parameters()) if technique["name"] != "L2 Регуляризация" \
            else optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device), optimizer=optimizer)

        final_accuracy = max(history["test_accs"])
        weights_distribution = analyze_weights(model)

        results.append({
            "technique": technique["name"],
            "final_accuracy": final_accuracy,
            "weights": weights_distribution
        })

    regularization_test_plot(results)

def regularization_test_plot(results):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    sns.barplot(x=[res["technique"] for res in results], y=[res["final_accuracy"] for res in results], ax=axes[0])
    axes[0].set_title("Финальная точность по технике регуляризации")
    axes[0].set_ylabel("Точность (%)")

    sns.boxplot(data=[res["weights"] for res in results], ax=axes[1])
    axes[1].set_xticklabels([res["technique"] for res in results], rotation=45)
    axes[1].set_title("Распределение весов по технике регуляризации")
    plt.tight_layout()
    plt.show()

if __name__ == "1__main__":
    regularization_test()

### 3.2 Адаптивная регуляризация (10 баллов)
# Реализуйте адаптивные техники:
# - Dropout с изменяющимся коэффициентом
# - BatchNorm с различными momentum
# - Комбинирование нескольких техник
# - Анализ влияния на разные слои сети

class AdaptiveDropout(torch.nn.Module):
    def __init__(self, initial_rate=0.5, min_rate=0.1, steps=1000):
        super().__init__()
        self.rate = initial_rate
        self.min_rate = min_rate
        self.steps = steps
        self.current_step = 0

    def forward(self, x):
        current_rate = max(self.rate * (1 - self.current_step / self.steps), self.min_rate)
        self.current_step += 1
        return F.dropout(x, p=current_rate, training=self.training)

class AdaptiveBatchNorm(torch.nn.BatchNorm1d):
    def __init__(self, num_features, initial_momentum=0.1, final_momentum=0.9, steps=1000):
        super().__init__(num_features=num_features, momentum=initial_momentum)
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.steps = steps
        self.current_step = 0

    def forward(self, x):
        current_momentum = self.initial_momentum + (self.final_momentum - self.initial_momentum) * (self.current_step / self.steps)
        self.momentum = current_momentum
        self.current_step += 1
        return super().forward(x)

class RegularizedFCN(FCN):
    def __init__(self, input_size, num_classes, layers):
        super().__init__(None, input_size, num_classes, layers=layers)
        self.layers = torch.nn.Sequential(*[
            self._convert_layer(layer) for layer in layers
        ])

    def _convert_layer(self, layer_spec):
        if isinstance(layer_spec, dict):
            if layer_spec["type"] == "linear":
                return torch.nn.Linear(layer_spec["in"], layer_spec["size"])
            elif layer_spec["type"] == "relu":
                return torch.nn.ReLU()
            elif layer_spec["type"] == "dropout":
                return AdaptiveDropout(initial_rate=layer_spec["rate"])
            elif layer_spec["type"] == "batch_norm":
                return AdaptiveBatchNorm(layer_spec["features"])
        else:
            raise ValueError(f"Unknown layer specification: {layer_spec}")

# Базовая конфигурация модели
base_model_config = [
    {"type": "linear", "in": 784, "size": 512},
    {"type": "relu"},
    {"type": "dropout", "rate": 0.3},
    {"type": "linear", "in": 512, "size": 256},
    {"type": "relu"},
    {"type": "batch_norm", "features": 256},
    {"type": "linear", "in": 256, "size": 128},
    {"type": "relu"},
    {"type": "linear", "in": 128, "size": 10}
]

def adaptive_regularization():
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)

    # Экспериментальная настройка с регуляризацией
    model = RegularizedFCN(input_size=784, num_classes=10, layers=base_model_config).to(device)

    # Оптимизация с L2 регуляризацией
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device), optimizer=optimizer)

    print(f"Final Test Accuracy: {max(history['test_accs']):.4f}%")

    plot_training_history(history)

    # Распределение весов модели
    weights = []
    for p in model.parameters():
        if p.dim() > 1:  # Выбираем только веса
            weights.extend(p.detach().cpu().numpy().flatten())

    adaptive_regularization_plot(weights)

def adaptive_regularization_plot(weights):
    sns.histplot(weights, bins=50, kde=True)
    plt.title("Distribution of Weights after Training")
    plt.show()

if __name__ == "__main__":
    adaptive_regularization()