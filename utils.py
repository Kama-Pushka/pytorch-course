import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
               best_test_loss: float, best_test_acc: float):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_test_loss': best_test_loss,
        'best_test_acc': best_test_acc,
    }
    torch.save(state_dict, path)


def load_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['epoch'], state_dict['best_test_loss'], state_dict['best_test_acc']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_first_batch(dataloader):
    """ Визуализирует первый батч изображений из DataLoader с соответствующими метками.
    :param dataloader: Экземпляр DataLoader """
    # Получаем первый батч из dataloader
    images, labels = next(iter(dataloader))

    n_samples = len(images)
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    n_cols = int(np.ceil(n_samples / n_rows))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 12))
    axes = axes.flatten()  # Приводим двумерный массив осей к плоскому виду

    # Показываем каждую картинку из батча
    for i in range(n_samples):
        image_np = images[i].numpy().transpose((1, 2, 0))  # CHW -> HWC
        image_np = np.clip(image_np, 0, 1)  # Корректировка диапазона пикселей

        ax = axes[i]
        ax.imshow(image_np)
        ax.set_title(f"Class: {'Positive' if labels[i].item() else 'Negative'}")
        ax.axis('off')

    # скрываем оставшиеся незадействованные ячейки (если batch_size не квадратичный)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_validation_curves(metrics):
    """ Визуализирует кривые обучения и валидации.
    :param metrics: Словарь с историей обучения и валидации """

    df_train = pd.DataFrame({"Epoch": range(1, len(metrics["train_loss"]) + 1),
                             "Type": ["Train"] * len(metrics["train_loss"]),
                             "Loss": metrics["train_loss"],
                             "Accuracy": metrics["train_acc"]})

    df_val = pd.DataFrame({"Epoch": range(1, len(metrics["val_loss"]) + 1),
                           "Type": ["Val"] * len(metrics["val_loss"]),
                           "Loss": metrics["val_loss"],
                           "Accuracy": metrics["val_acc"]})
    df_combined = pd.concat([df_train, df_val]).reset_index(drop=True)

    # ГРАФИК ПОТЕРЬ
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df_combined, x="Epoch", y="Loss", hue="Type", style="Type", markers=True)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(title="Type")

    # ГРАФИК ТОЧНОСТИ
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df_combined, x="Epoch", y="Accuracy", hue="Type", style="Type", markers=True)
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(title="Type")

    plt.tight_layout()
    plt.show()


def compute_confusion_matrix(model, dataloader, device):
    """ Рассчитывает и отображает confusion matrix для модели.
    :param model: PyTorch модель
    :param dataloader: DataLoader для тестирования
    :param device: Устройство (GPU/CUDA)"""
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_targets.extend(labels.cpu().numpy())  # Метки
            all_predictions.extend(predictions.cpu().numpy())  # Предсказанные классы

    cm = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return cm