import torch
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(history['train_losses'], label='Train loss')
    ax1.plot(history['test_losses'], label='Test loss')
    ax1.set_title('Loss')
    ax1.legeng()

    ax2.plot(history['train_accs'], label='Train acc')
    ax2.plot(history['test_accs'], label='Test acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def compare_models(fc_history, cnn_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax1.set_title('Test Loss Comparison')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()