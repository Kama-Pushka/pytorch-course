import torch
import torch.nn as nn
from tqdm import tqdm

from utils import save_model


def run_epoch(model, data_loader, loss_fn, optimizer=None, device=None, is_test=False):
    """ Выполняет одну эпоху обучения или валидации.
    :param model: Инстанс PyTorch модели
    :param data_loader: DataLoader (для обучения или тестирования)
    :param loss_fn: Критерий потерь (например, BCELoss)
    :param optimizer: Оптимизатор (только для режима обучения)
    :param device: Устройство (CPU/GPU)
    :param is_test: Флаг, определяющий режим (training vs testing)
    :return: Средняя потеря и точность за эпоху """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Переключаемся между режимами обучения и тестирования
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(not is_test):
        progress_bar = tqdm(data_loader, leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.long().to(device)

            if not is_test:
                optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = loss_fn(outputs, labels)

            if not is_test:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            total_correct += (pred == labels).float().sum().item()
            total_samples += inputs.size(0)

            progress_bar.set_description(
                f"Loss: {total_loss / total_samples:.4f}, Acc: {total_correct / total_samples:.4f}")

    return total_loss / len(data_loader), total_correct / total_samples


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device=None, optimizer=None):
    """ Управляет процессом обучения и периодической проверкой модели.
    :param model: Инстанс PyTorch модели
    :param train_loader: DataLoader для обучения
    :param test_loader: DataLoader для тестирования
    :param epochs: Количество эпох обучения
    :param lr: Скорость обучения
    :param device: Устройство (CPU/GPU)
    :param optimizer: Пользовательский оптимизатор (опциональный)
    :return: Словарь метрик обучения и тестирования """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if optimizer is None:
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss
    metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_test_loss = float('inf')
    best_test_acc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}:")

        # Обучение на одной эпохе
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, is_test=False)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Валидация на тестовом наборе
        val_loss, val_acc = run_epoch(model, test_loader, loss_fn, device=device, is_test=True)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

        if val_loss < best_test_loss:
            best_test_loss = val_loss
            save_model('params/best_loss.pth', model, optimizer, epoch, best_test_loss, best_test_acc)

        if val_acc > best_test_acc:
            best_test_loss = val_acc
            save_model('params/best_acc.pth', model, optimizer, epoch, best_test_loss, best_test_acc)

    return metrics