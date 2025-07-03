import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import Dataset
from homework_model_modification import LinearRegressionTorch, train_linear_model, LogisticRegressionTorch, \
    train_logistic_model

### 2.1 Кастомный Dataset класс (15 баллов)
# Создайте кастомный класс датасета для работы с CSV файлами:
# - Загрузка данных из файла
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)

class CSVDataset(Dataset):
    # Поддерживает два основных формата данных - категориальные и числовые

    def __init__(self, file_path, target_col, cat_cols=None, num_cols=None, normalize=True):
        """ Параметры:
        - file_path: путь к CSV файлу
        - target_col: название колонки с целевыми значениями
        - cat_cols: список категориальных признаков
        - num_cols: список числовых признаков
        - normalize: флаг нормализации числовых признаков
        """
        self.dataframe = pd.read_csv(file_path, dtype={
            target_col: float,
            **({col: object for col in cat_cols}),  # Преобразуем категориальные признаки в object
            **({col: float for col in num_cols})  # Преобразуем числовые признаки в float
        })

        # Названия признаков и целевого столбца
        self.target_col = target_col
        self.features = self.dataframe.columns.difference([target_col])

        self._preprocess(cat_cols, num_cols, normalize)

    def _preprocess(self, cat_cols, num_cols, normalize):
        """Выполнение предобработки данных."""
        # Нормализация числовых признаков
        if normalize and num_cols:
            scaler = MinMaxScaler()
            self.dataframe[num_cols] = scaler.fit_transform(self.dataframe[num_cols])

        # Кодирование категориальных признаков
        if cat_cols:
            le = LabelEncoder()
            for col in cat_cols:
                test = le.fit_transform(self.dataframe[col])
                self.dataframe[col] = test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        features = torch.tensor(row[self.features].values, dtype=torch.float32)
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        return features, target

### 2.2 Эксперименты с различными датасетами (15 баллов)
# Найдите csv датасеты для регрессии и бинарной классификации и, применяя наработки из предыдущей части задания, обучите линейную и логистическую регрессию

def linreg():
    train_dataset = CSVDataset("data/insurance.csv", "charges", ["children", "region", "sex", "smoker"],
                               ["age", "bmi", "charges"])
    val_dataset = CSVDataset("data/insurance_val.csv", "charges", ["children", "region", "sex", "smoker"],
                             ["age", "bmi", "charges"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = LinearRegressionTorch(6)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_linear_model(model, loss_fn, optimizer, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), 'linreg_torch.pth')


def logreg():
    train_dataset = CSVDataset("data/diabetes.csv", "Outcome", [],
                               ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
    val_dataset = CSVDataset("data/diabetes_val.csv", "Outcome", [],
                             ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = LogisticRegressionTorch(8, 2)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_logistic_model(model, loss_fn, optimizer, train_dataloader, val_dataloader, epochs=200)
    torch.save(model.state_dict(), 'logreg_torch.pth')

if __name__ == "__main__":
    linreg()
    logreg()
