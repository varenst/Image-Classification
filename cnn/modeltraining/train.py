import torch
import torch.nn as nn
import torch.optim as optim
from cnn.cnn_setup import Net
from cnn.datasetloader import load_data

def train():
    # doing this incase you want to test if I wrote this correctly
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

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    torch.save(net.state_dict(), "catdog_cnn.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
