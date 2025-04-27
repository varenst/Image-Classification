import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from cnn.cnn_setup import Net
from cnn.datasetloader import load_data

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def show_misclassified(test_loader, model, classes, max_images=10, device='cpu', save_path=None):
    model.eval()
    misclassified = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append(inputs[i].cpu())
                    true_labels.append(labels[i].cpu().item())
                    pred_labels.append(preds[i].cpu().item())

    total_misclassified = len(misclassified)
    num_images = min(max_images, total_misclassified)

    if num_images == 0:
        print("No misclassified images found.")
        return

    grid_size = int(num_images ** 0.5) + 1

    plt.figure(figsize=(15, 15))
    for idx in range(num_images):
        img = misclassified[idx] * 0.5 + 0.5  # Unnormalize
        img = img.permute(1, 2, 0).numpy()

        plt.subplot(grid_size, grid_size, idx + 1)
        plt.imshow(img)
        plt.title(f"True: {classes[true_labels[idx]]}\nPred: {classes[pred_labels[idx]]}", color="red")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, classes = load_data(
        train_dir="./data",
        test_dir="./testdata"
    )

    net = Net().to(device)
    net.load_state_dict(torch.load("catdog_cnn.pth"))
    net.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds, target_names=classes))
    plot_confusion_matrix(cm, classes)
    show_misclassified(test_loader, net, classes, max_images=10, device=device)

if __name__ == "__main__":
    test()
