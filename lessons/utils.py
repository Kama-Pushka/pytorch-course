import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

### В PyTorch для своих целей нужно определять свои версии Dataset
### Для своего датасета достаточно переопределить данные методы:
# __len__(self), __getitem__(self, idx)
### ну то есть чтобы DataLoader мог выгрузить из нашего датасета данные

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index): # обязательно к оптимизации! ибо используется постоянно!
        return self.X[index], self.y[index] # стандартный collator спокойно это обрабатывает (если возвращаем как-то по другому, то можно определить collator: batch -> tensor_batch)

def make_regression_data(n=100, noise=0.2, source='random'):
    if source=='random':
        X = torch.randn(n, 1)
        w, b = -5, 10
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source=='diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1) # вектор-столбец (n, 1)
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = torch.tensor(data['data'], dtype=torch.float32)
    y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
    return X, y

def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true) ** 2).mean()

def log_epoch(epoch, avg_loss, **metrics):
    message = f'Epoch: {epoch}\tloss: {avg_loss:.4f}'
    for k, v in metrics.items():
        message += f'\t{k}: {v:.4f}'
    print(message)

# нормализация на (0, 1)
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (torch.exp(-x) + 1)

# только для бинарной классификации
def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred > 0.5).float() # вычисляем количество единичек
    return (y_pred_bin == y_true).float().mean().item() # считаем количество верных ответов и превращаем его в число (а не тензор)

def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()