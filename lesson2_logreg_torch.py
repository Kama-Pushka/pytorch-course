import torch
from utils import CustomDataset, log_epoch, accuracy, make_classification_data


class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear.forward(x)


def logistic_regression_torch_test():
    EPOCHS = 100

    X, y = make_classification_data()
    dataset = CustomDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    lr = 0.1

    model = LogisticRegressionTorch(X.shape[1])
    loss_fn = torch.nn.BCEWithLogitsLoss() # сигмоида применяется тут автоматически
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0

        for i, (batch_x, batch_y) in enumerate(dataloader): # tqdm(dataloader)
            optimizer.zero_grad() # чтобы не накапливался градиент, необходим для большого батча

            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy(torch.sigmoid(y_pred), batch_y)

        avg_loss = epoch_loss / len(dataset)
        avg_acc = epoch_acc / len(dataloader)
        if epoch % 1 == 0:
            log_epoch(epoch, avg_loss, accuracy=avg_acc)
    print(model.linear.weight.data, model.linear.bias.data)


if __name__ == "__main__":
    logistic_regression_torch_test()