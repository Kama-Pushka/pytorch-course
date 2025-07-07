import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from lessons.lesson5_datasets import CustomImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Размер, подходящий для ResNet18
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Нормализация по стандартам ImageNet
])

train_dataset = CustomImageDataset('data/train/', transform=transform)
val_dataset = CustomImageDataset('data/test/', transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

num_classes = len(train_dataset.get_class_names())
model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(device)  # Меняем последний слой

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    history['train_loss'].append(train_loss)

    model.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()
    test_loss = val_running_loss / len(val_loader)
    history['val_loss'].append(test_loss)
    test_accuracy = correct / total
    history['val_accuracy'].append(test_accuracy)

    print(f'Epoch {epoch + 1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train loss')
plt.plot(history['val_loss'], label='Test loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_accuracy'])
plt.title('Accuracy in Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()