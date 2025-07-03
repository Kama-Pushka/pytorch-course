import torch
from utils import CustomDataset, make_regression_data, mse, log_epoch
from tqdm import tqdm # для прогрессбара

### Линейная регрессия
# решает задачу регрессии
# написана по математическим формулам

class LinearRegression:
    def __init__(self, in_features): # кол-во параметров
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return X @ self.w + self.b

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


def linear_regression_test():
    # random
    EPOCHS = 100

    X, y = make_regression_data(1000)
    dataset = CustomDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    model = LinearRegression(1)  # потому что random генерирует только одномерную последовательность -> один параметр
    lr = 0.1 # learning_rate - коррекция весов (как сильно будут изменяться веса за одну эпоху)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0

        for i, (batch_x, batch_y) in enumerate(dataloader): # tqdm(dataloader)
            y_pred = model.forward(batch_x)
            loss = mse(y_pred, batch_y)
            epoch_loss += loss.item()  # Чтобы loss не был тензором, добавляем item()

            model.backward(batch_x, batch_y, y_pred)
            model.step(lr)
        avg_loss = epoch_loss / len(dataloader)
        if epoch % 1 == 0:
            log_epoch(epoch, avg_loss)
    print(model.w, model.b)  # для random ~-5, ~10, они же являются исходными -> лин регрессия работает корректно


if __name__ == "__main__":
    linear_regression_test()