import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class CardiomegalyDataset(Dataset):
    def __init__(self, data_dir: str, image_size: (int, int), recursive=True):
        """
        Инициализирует набор данных кардиомегалии с возможностью рекурсивного поиска классов.
        :param data_dir: Основной путь к директории с изображениями ('train' or 'test')
        :param image_size: Целевой размер изображения после преобразования (width, height)
        :param recursive: Рекурсивно искать классы в подпапках (default: True)
        """
        self.image_size = image_size
        self.transform = Compose([Resize(image_size), ToTensor()])

        # Найдем все папки 'true' и 'false'
        root_path = Path(data_dir)
        if recursive:
            # Рекурсивно ищем папки 'true' и 'false'
            true_folders = list(root_path.rglob("true"))
            false_folders = list(root_path.rglob("false"))
        else:
            # Если не рекурсивно, ограничимся первым уровнем вложенности
            true_folders = list((root_path / "true").iterdir())
            false_folders = list((root_path / "false").iterdir())

        # Собираем полные пути к файлам и соответствующим меткам
        paths_and_labels = []
        for folder in true_folders:
            files_in_true = sorted(folder.iterdir())
            paths_and_labels.extend([(file, True) for file in files_in_true])

        for folder in false_folders:
            files_in_false = sorted(folder.iterdir())
            paths_and_labels.extend([(file, False) for file in files_in_false])

        self.paths_and_labels = paths_and_labels

    def __len__(self):
        return len(self.paths_and_labels)

    def __getitem__(self, idx):
        path, label = self.paths_and_labels[idx]
        img = Image.open(path).convert('RGB')
        transformed_img = self.transform(img)
        t = int(label)
        target = torch.tensor(t)  # True/False превращаются в 1/0
        return transformed_img, target

def get_cardiomegaly_dataloaders(
    train_data_path: str,
    test_data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle_train: bool = True,
    recursive: bool = True
):
    """
    Создаем загрузчики данных для тренировочных и тестовых данных, используя CardiomegalyDataset.
    :param train_data_path: Путь к тренировочным данным
    :param test_data_path: Путь к тестовым данным
    :param batch_size: Размер батчей
    :param num_workers: Число потоков для параллельной загрузки данных
    :param shuffle_train: Нужно ли перемешивать тренировочные данные
    :param recursive: Флаг, разрешающий рекурсивный поиск классов (по умолчанию включен)
    :return: Кортеж (train_loader, test_loader) """

    # Инициализируем тренировочную и тестовую выборки с учётом рекурсии
    train_dataset = CardiomegalyDataset(train_data_path, image_size=(224, 224), recursive=recursive)
    test_dataset = CardiomegalyDataset(test_data_path, image_size=(224, 224), recursive=recursive)

    # Создаем объект DataLoader для тренировок
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )

    # Создаем объект DataLoader для тестов
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader