import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from cnn.cnn_setup import Net
from cnn.datasetloader import load_data

PATH = './cat_dog_model.pth'

def test():
    trainloader, testloader, classes = load_data("./traindata")

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 40 test images: {100 * correct // total} %"
    )

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accuracy = 0.0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

