import torch
import torch.nn as nn
import torch.optim as optim
from cnn.cnn_setup import Net
from cnn.datasetloader import load_data

import matplotlib.pyplot as plt

def plot_training(train_losses, train_accuracies):
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'r', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'b', label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train():
    # doing this incase you want to test, if my code works correctly
    # I ran this on my cpu so it was slower but still worked
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = load_data(
        train_dir="./data",
        test_dir="./testdata"
    )

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 20
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        
        net.train()
        running_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    torch.save(net.state_dict(), "catdog_cnn.pth")
    print("Model saved.")
    
    plot_training(train_losses, train_accuracies)

if __name__ == "__main__":
    train()
