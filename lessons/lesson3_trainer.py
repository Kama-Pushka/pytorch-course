import torch
from tqdm import tqdm

from lessons.lesson3_datasets import get_mnist_dataloaders
from lessons.lesson3_models import FCN


def save_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
               best_test_loss: float, best_test_acc: float):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_test_loss': best_test_loss,
        'best_test_acc': best_test_acc,
    }
    torch.save(state_dict, path)


def load_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['epoch'], state_dict['best_test_loss'], state_dict['best_test_acc']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_epoch(model, data_loader, loss_fn, optimizer=None, device='cuda', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    total_correct = 0
    total = 0

    for i, (image, target) in enumerate(tqdm(data_loader)):
        image, target = image.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model.forward(image)
        loss = loss_fn(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        total_correct += (pred == target).float().sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), total_correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda', optimizer = None):
    loss_fn = torch.nn.CrossEntropyLoss()
    if optimizer is None: optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_test_loss = float('inf')
    best_test_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, False)
        test_loss, test_acc = run_epoch(model, test_loader, loss_fn, None, device, True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            #save_model('../params/best_loss.pth', model, optimizer, epoch, best_test_loss, best_test_acc)

        if test_acc > best_test_acc:
            best_test_loss = test_loss
            #save_model('../params/best_acc.pth', model, optimizer, epoch, best_test_loss, best_test_acc)


        print(f'Epoch {epoch + 1}/{epochs}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f},')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


if __name__ == "__main__":
    config = {
        'input_size': 784,
        'num_classes': 10,
        'layers': [
            {'type': 'linear', 'size': 512},
            {'type': 'relu'},
            {'type': 'linear', 'size': 256},
            {'type': 'relu'},
            {'type': 'linear', 'size': 128},
            {'type': 'relu'},
        ]
    }

    model = FCN(**config)
    print(model)

    print(f'Model params: {count_parameters(model)}')

    train_dl, test_dl = get_mnist_dataloaders(batch_size=1024)

    train_model(model, train_dl, test_dl, epochs=10, lr=0.001, device='cuda')