import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchsummary import summary
import cv2


def has_cuda():
    return torch.cuda.is_available()


def which_device():
    return torch.device("cuda" if has_cuda() else "cpu")


def init_seed(args):
    torch.manual_seed(args.seed)

    if has_cuda():
        print("CUDA Available")
        torch.cuda.manual_seed(args.seed)


def show_model_summary(model, input_size):
    print(summary(model, input_size=input_size))


def imshow(img):
    img = denormalize(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def normalize(tensor, mean=[0.4914, 0.4822, 0.4465],
              std=[0.2023, 0.1994, 0.2010]):
    single_img = False
    if tensor.ndimension() == 3:
        single_img = True
        tensor = tensor[None, :, :, :]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.sub(mean).div(std)
    return ret[0] if single_img else ret


def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]):
    single_img = False
    if tensor.ndimension() == 3:
        single_img = True
        tensor = tensor[None, :, :, :]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret


def truth_checker(model, loader, device, criterion):
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return test_loss, correct/total


def plot_images(img_data, classes, img_name):
    figure = plt.figure(figsize=(10, 10))

    num_of_images = len(img_data)
    for index in range(1, num_of_images + 1):
        img = img_data[index-1]["img"]  # unnormalize
        plt.subplot(5, 5, index)
        plt.axis('off')
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.title("Predicted: %s\nActual: %s" % (
            classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))

    plt.tight_layout()
    plt.savefig(img_name)


def plot_graph(data, metric):
    fig, ax = plt.subplots()

    for sub_metric in data.keys():
        ax.plot(data[sub_metric], label=sub_metric)

    plt.title(f'Change in %s' % (metric))
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    ax.legend()
    plt.show()

    fig.savefig(f'%s_change.png' % (metric.lower()))
