import torchvision
import torchvision.transforms as tf
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

## Подгрузка картинок для обучения

class MNISTDataset(Dataset):
    def __init__(self, train: bool = True):
        self.dataset = torchvision.datasets.MNIST(
            root='../data',
            train=train,
            download=True
        )
        self.transform = tf.ToTensor() # преобразование Pillow-картинки в тензор

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)

class CIFARDataset(Dataset):
    def __init__(self, train: bool = True):
        self.dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=train,
            download=True
        )
        self.transform = tf.ToTensor() # преобразование Pillow-картинки в тензор

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def get_mnist_dataloaders(batch_size: int = 128):
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def get_cifar_dataloaders(batch_size: int = 128):
    train_dataset = CIFARDataset(train=True)
    test_dataset = CIFARDataset(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader