import torch
from utils import make_regression_data, CustomDataset, log_epoch


class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear.forward(x)


def linear_regression_torch_test():
    EPOCHS = 100

    X, y = make_regression_data(1000)
    dataset = CustomDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    lr = 0.1  # learning_rate - коррекция весов (как сильно будут изменяться веса за одну эпоху)

    model = LinearRegressionTorch(1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0

        for i, (batch_x, batch_y) in enumerate(dataloader):  # tqdm(dataloader)
            optimizer.zero_grad() # чтобы не накапливался градиент, необходим для большого батча

            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        if epoch % 1 == 0:
            log_epoch(epoch, avg_loss)
    print(model.linear.weight.data, model.linear.bias.data)


if __name__ == "__main__":
    linear_regression_torch_test()