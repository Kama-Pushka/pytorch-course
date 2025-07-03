import torch
import seaborn as sns
import matplotlib.pyplot as plt
from utils import make_regression_data, CustomDataset, log_epoch, make_multiclass_classification_data
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

### 1.1 Расширение линейной регрессии (15 баллов)
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
# - Добавьте early stopping

class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear.forward(x)

def train_linear_model(model, loss_fn, optimizer, dataloader, val_dataloader, l1_lambda=0.01, l2_lambda=0.01, patience=5, epochs=100):
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        # Обучение модели
        model.train()
        for i, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model.forward(batch_x)
            loss = loss_fn.forward(y_pred, batch_y)

            # Применение L1 и L2 регуляризации
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            total_loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_train_loss = epoch_loss / len(dataloader)

        # Проверка качества на валидационном датасете
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (val_batch_x, val_batch_y) in enumerate(val_dataloader):
                val_pred = model.forward(val_batch_x)
                val_loss += loss_fn.forward(val_pred, val_batch_y).item()
        avg_val_loss = val_loss / len(val_dataloader)

        # Логика early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(f"Ранний стоп на эпохе {epoch}. avg_val_loss не улучшился.")
                break

        if epoch % 1 == 0:
            log_epoch(epoch, avg_train_loss, avg_val_loss=avg_val_loss)

    print(model.linear.weight.data, model.linear.bias.data)

if __name__ == "__main__":
    X, y = make_regression_data(1000)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = LinearRegressionTorch(1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_linear_model(model, loss_fn, optimizer, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), 'linreg_torch.pth')

### Расширение логистической регрессии (15 баллов)
# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# - Добавьте визуализацию confusion matrix

class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear.forward(x)


def train_logistic_model(model, loss_fn, optimizer, dataloader, val_dataloader, epochs=500):
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0

        # Обучение
        model.train()
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_y = batch_y.squeeze().long()

            optimizer.zero_grad()
            outputs = model.forward(batch_x)
            loss = loss_fn.forward(outputs, batch_y)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs, dim=1)
            epoch_loss += loss.item() * batch_x.size(0)
            epoch_acc += torch.sum(predicted == batch_y).item()

        avg_train_loss = epoch_loss / len(dataloader.dataset)
        avg_train_acc = epoch_acc / len(dataloader.dataset)

        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                batch_y = batch_y.squeeze().long()

                outputs = model.forward(batch_x)
                predicted = torch.argmax(outputs, dim=1)
                loss = loss_fn(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                correct += torch.sum(predicted == batch_y)
                all_targets.extend(batch_y.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader.dataset)
        avg_val_acc = correct / len(val_dataloader.dataset)

        report = classification_report(all_targets, all_predictions, output_dict=True)
        macro_avg_precision = report["macro avg"]["precision"]
        macro_avg_recall = report["macro avg"]["recall"]
        macro_avg_f1 = report["macro avg"]["f1-score"]

        try: # только для бинарной классификации
            auc = roc_auc_score(all_targets, all_predictions)
        except ValueError:
            auc = "Not applicable"

        if epoch % 1 == 0:
            log_epoch(epoch, avg_train_loss, avg_train_acc=avg_train_acc, avg_val_loss=avg_val_loss, accuracy=avg_val_acc,
                      precision=macro_avg_precision, recall=macro_avg_recall, f1=macro_avg_f1, auc=auc)

    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    print(model.linear.weight.data, model.linear.bias.data)


if __name__ == "__main__":
    X, y = make_multiclass_classification_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=30, shuffle=False,)

    num_classes = len(torch.unique(y))

    model = LogisticRegressionTorch(X.shape[1], num_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    train_logistic_model(model, loss_fn, optimizer, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), 'logreg_torch.pth')
