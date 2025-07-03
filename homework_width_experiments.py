import time
import torch
from utils.lesson3_datasets import get_mnist_dataloaders
from utils.lesson3_models import FCN
from utils.lesson3_trainer import count_parameters, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 2.1 Сравнение моделей разной ширины (15 баллов)
# Создайте модели с различной шириной слоев:
# - Узкие слои: [64, 32, 16]
# - Средние слои: [256, 128, 64]
# - Широкие слои: [1024, 512, 256]
# - Очень широкие слои: [2048, 1024, 512]
#
# Для каждого варианта:
# - Поддерживайте одинаковую глубину (3 слоя)
# - Сравните точность и время обучения
# - Проанализируйте количество параметров

def width_test():
    configs = [
        {
            "name": "Узкие слои",
            "layers": [
                {"type": "linear", "size": 64},
                {"type": "relu"},
                {"type": "linear", "size": 32},
                {"type": "relu"},
                {"type": "linear", "size": 16},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "Средние слои",
            "layers": [
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "linear", "size": 128},
                {"type": "relu"},
                {"type": "linear", "size": 64},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "Широкие слои",
            "layers": [
                {"type": "linear", "size": 1024},
                {"type": "relu"},
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 256},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        },
        {
            "name": "Очень широкие слои",
            "layers": [
                {"type": "linear", "size": 2048},
                {"type": "relu"},
                {"type": "linear", "size": 1024},
                {"type": "relu"},
                {"type": "linear", "size": 512},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        }
    ]

    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)

    for cfg in configs:
        print(f"\nTraining model with configuration: {cfg['name']}\n")

        start_time = time.time()

        model = FCN(input_size=784, num_classes=10, layers=cfg["layers"]).to(device)
        params_count = count_parameters(model)
        print(f"Number of parameters: {params_count}")
        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Time taken to train: {training_time:.2f} seconds")

        # plot_training_history(history)


if __name__ == "__main__":
    width_test()

### 2.2 Оптимизация архитектуры (10 баллов)
# Найдите оптимальную архитектуру:
# - Используйте grid search для поиска лучшей комбинации
# - Попробуйте различные схемы изменения ширины (расширение, сужение, постоянная)
# - Визуализируйте результаты в виде heatmap

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ParameterGrid


def create_layers_config(widths):
    """Создает конфигурацию слоев"""
    return [
        {"type": "linear", "size": widths[0]}, {"type": "relu"},
        {"type": "linear", "size": widths[1]}, {"type": "relu"},
        {"type": "linear", "size": widths[2]}, {"type": "relu"},
        {"type": "linear", "size": 10}
    ]


def evaluate_architecture(train_loader, test_loader, width_scheme):
    """ Оценивает одну архитектуру по точности и количеству параметров. Возвращает словарь с результатами """
    start_time = time.time()

    model = FCN(input_size=784, num_classes=10, layers=create_layers_config(width_scheme)).to(device)
    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

    accuracy = max(history["test_accs"])
    training_time = time.time() - start_time
    param_count = count_parameters(model)

    return {
        "width_scheme": tuple(width_scheme),  # преобразуем список в кортеж!
        "accuracy": accuracy,
        "param_count": param_count,
        "training_time": training_time
    }


def search_best_model():
    train_loader, test_loader = get_mnist_dataloaders(batch_size=1024)

    parameter_grid = {
        "width_scheme": [
            [64, 128, 256],  # Расширяющаяся схема
            [256, 128, 64],  # Сужающаяся схема
            [128, 128, 128],  # Постоянная ширина
            [256, 256, 256],
            [512, 512, 512],
            [1024, 1024, 1024]
        ]
    }
    results = []

    # Запускаем grid search
    for params in ParameterGrid(parameter_grid):
        result = evaluate_architecture(train_loader, test_loader, params["width_scheme"])
        results.append(result)
        print(
            f"Evaluated architecture: {result['width_scheme']}. Accuracy: {result['accuracy']:.4f}, Time: {result['training_time']:.2f}s.")

    df_results = pd.DataFrame(results)
    pivot_table = df_results.pivot(index='width_scheme', columns='param_count', values='accuracy')
    best_result = df_results.sort_values(by="accuracy", ascending=False).iloc[0]
    print("\nBest Architecture:")
    print(best_result)

    # Heatmap результатов (точность vs. кол-во параметров)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".4f")
    plt.title("Accuracy by Width Scheme and Number of Parameters")
    plt.show()


if __name__ == "__main__":
    search_best_model()