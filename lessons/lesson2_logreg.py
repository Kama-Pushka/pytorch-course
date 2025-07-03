import torch
from utils import CustomDataset, log_epoch, accuracy, make_classification_data, sigmoid

### Логистическая регрессия
# решает задачу классификации
# написана по математическим формулам

class LogisticRegression:
    def __init__(self, in_features): # кол-во параметров
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return sigmoid(X @ self.w + self.b)

    def forward(self, X):
        return self.__call__(X)

    def backward(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        # y_pred = self.forward(X)
        n = X.size(0)
        self.dw = -1 / n * X.T @ (y - y_pred)
        self.db = -(y - y_pred).mean() # -1 / n * (y - y_pred)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


def logistic_regression_test():
    EPOCHS = 100

    X, y = make_classification_data()
    dataset = CustomDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    model = LogisticRegression(X.shape[1]) # сколько признаков - столько весов
    lr = 0.1

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0

        for i, (batch_x, batch_y) in enumerate(dataloader): # tqdm(dataloader)
            y_pred = model.forward(batch_x)
            loss = -(batch_y * torch.log(y_pred + 1e-8) + (1 - batch_y) * torch.log((1 - y_pred) + 1e-8)).mean() # бинарный cross-entropy, +1e-8 чтобы log не расходился в нуле
            epoch_loss += loss.item()  # Чтобы loss не был тензором, добавляем item()
            acc = accuracy(y_pred, batch_y)
            epoch_acc += acc

            model.backward(batch_x, batch_y, y_pred)
            model.step(lr)
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        if epoch % 1 == 0:
            log_epoch(epoch, avg_loss, accuracy=avg_acc)
    print(model.w, model.b) # выводим веса


if __name__ == "__main__":
    logistic_regression_test()