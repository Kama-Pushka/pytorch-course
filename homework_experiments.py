import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from homework_model_modification import LinearRegressionTorch, train_linear_model
from utils import make_regression_data, CustomDataset

### 3.1 Исследование гиперпараметров (10 баллов)
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц

# Функция для сбора статистики весов
def collect_weight_history(model, loss_fn, optimizer, dataloader, epochs=100):
    history_weights = []
    for _ in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(y_pred, batch_y)
            loss.backward()
            optimizer.step()
        history_weights.append(model.linear.weight.data.clone().item())  # Сохраняем текущее значение веса
    return history_weights

# Генерация данных
X, y = make_regression_data(1000)

# Датасеты
train_dataset = CustomDataset(X, y)

def learning_rate_test():
    # Динамическое формирование загрузчиков
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Эксперимент с Learning Rates
    lrs = [0.001, 0.01, 0.1]
    histories_lr = {}

    for lr in lrs:
        model = LinearRegressionTorch(1)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        histories_lr[f'LR={lr}'] = collect_weight_history(model, loss_fn, optimizer, train_dataloader)

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    for key, hist in histories_lr.items():
        plt.plot(hist, label=key)
    plt.title('Изменение весов при разных Learning Rates')
    plt.xlabel('Эпоха')
    plt.ylabel('Вес')
    plt.legend()
    plt.grid(True)
    plt.show()

def batch_size_test():
    # Эксперимент с Batch Sizes
    batch_sizes = [32, 128, 256]
    histories_bs = {}

    for bs in batch_sizes:
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

        model = LinearRegressionTorch(1)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        histories_bs[f'BS={bs}'] = collect_weight_history(model, loss_fn, optimizer, train_dataloader)

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    for key, hist in histories_bs.items():
        plt.plot(hist, label=key)
    plt.title('Изменение весов при разных Batch Sizes')
    plt.xlabel('Эпоха')
    plt.ylabel('Вес')
    plt.legend()
    plt.grid(True)
    plt.show()

def optimizer_test():
    # Эксперимент с оптимизаторами
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSProp': torch.optim.RMSprop
    }
    histories_opt = {}

    for opt_name, OptimizerCls in optimizers.items():
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        model = LinearRegressionTorch(1)
        loss_fn = torch.nn.MSELoss()
        optimizer = OptimizerCls(model.parameters(), lr=0.01)
        histories_opt[opt_name] = collect_weight_history(model, loss_fn, optimizer, train_dataloader)

    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    for key, hist in histories_opt.items():
        plt.plot(hist, label=key)
    plt.title('Изменение весов при разных оптимизаторах')
    plt.xlabel('Эпоха')
    plt.ylabel('Вес')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    learning_rate_test()
    batch_size_test()
    optimizer_test()

### 3.2 Feature Engineering (10 баллов)
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from utils import make_regression_data

def baseline_model():
    # Базовая модель
    model = LinearRegressionTorch(X_train_tensor.shape[1])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Обучение базовой модели
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    val_dataset = CustomDataset(X_val_tensor, y_val_tensor)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    train_linear_model(model, loss_fn, optimizer, train_dataloader, val_dataloader)

    return model

def expanded_model():
    ## Feature engineering

    # Полиномиальные признаки
    poly_feat = PolynomialFeatures(degree=3, include_bias=False)
    poly_features_train_tensor = torch.tensor(poly_feat.fit_transform(X_train), dtype=torch.float32)
    poly_features_val_tensor = torch.tensor(poly_feat.transform(X_val), dtype=torch.float32)

    # Взаимодействия признаков
    interaction_features_train = torch.zeros(
        (X_train_tensor.shape[0], X_train_tensor.shape[1] * X_train_tensor.shape[1]))
    interaction_features_val = torch.zeros((X_val_tensor.shape[0], X_val_tensor.shape[1] * X_val_tensor.shape[1]))

    for i in range(X_train_tensor.shape[1]):
        for j in range(X_train_tensor.shape[1]):
            interaction_features_train[:, i * X_train_tensor.shape[1] + j] = X_train_tensor[:, i] * X_train_tensor[:, j]
            interaction_features_val[:, i * X_val_tensor.shape[1] + j] = X_val_tensor[:, i] * X_val_tensor[:, j]

    # Статистические признаки
    mean_features_train = torch.mean(X_train_tensor, dim=0, keepdim=True).expand(X_train_tensor.shape[0], -1)
    std_features_train = torch.std(X_train_tensor, dim=0, keepdim=True).expand(X_train_tensor.shape[0], -1)

    mean_features_val = torch.mean(X_val_tensor, dim=0, keepdim=True).expand(X_val_tensor.shape[0], -1)
    std_features_val = torch.std(X_val_tensor, dim=0, keepdim=True).expand(X_val_tensor.shape[0], -1)

    # Объединяем признаки
    final_X_train = torch.cat(
        [X_train_tensor, poly_features_train_tensor, interaction_features_train, mean_features_train,
         std_features_train], dim=1)
    final_X_val = torch.cat(
        [X_val_tensor, poly_features_val_tensor, interaction_features_val, mean_features_val, std_features_val], dim=1)

    # Новый линейный регрессор с увеличенным числом признаков
    expanded = LinearRegressionTorch(final_X_train.shape[1])
    loss_fn_expanded = torch.nn.MSELoss()
    optimizer_expanded = torch.optim.SGD(expanded.parameters(), lr=0.1)

    # Обучение расширенной модели
    train_dataset_expanded = CustomDataset(final_X_train, y_train_tensor)
    val_dataset_expanded = CustomDataset(final_X_val, y_val_tensor)

    train_dataloader_expanded = torch.utils.data.DataLoader(train_dataset_expanded, batch_size=128, shuffle=True)
    val_dataloader_expanded = torch.utils.data.DataLoader(val_dataset_expanded, batch_size=128, shuffle=False)

    train_linear_model(expanded, loss_fn_expanded, optimizer_expanded, train_dataloader_expanded,
                       val_dataloader_expanded)

    return expanded, final_X_val

if __name__ == "__main__":
    X, y = make_regression_data(source='diabetes')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    model = baseline_model()
    expanded, final_X_val = expanded_model()

    # Оценка базовой модели
    with torch.no_grad():
        baseline_preds = model(X_val_tensor)
        baseline_mse = torch.nn.MSELoss()(baseline_preds, y_val_tensor).item()

    # Оценка расширенной модели
    with torch.no_grad():
        expanded_preds = expanded(final_X_val)
        expanded_mse = torch.nn.MSELoss()(expanded_preds, y_val_tensor).item()

    print(f"Модель с оригинальными признаками (Baseline Model):\nMSE = {baseline_mse:.4f}")
    print(f"Расширенная модель (Expanded Features):\nMSE = {expanded_mse:.4f}")