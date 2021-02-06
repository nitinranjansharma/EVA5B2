import torch
#import torch.nn.functional as F
from tqdm import tqdm


def train(trainloader, device, optimizer, net, criterion):
    net.train()
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    loss = 0
    for i, data in enumerate(pbar):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # get the index of the max log-probability
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={i} Accuracy={100*correct/processed:0.2f}')


def test(testloader, device, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def training(EPOCHS, train, test, testloader, device, net, optimizer, criterion, trainloader):
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(trainloader, device, optimizer, net, criterion)
        test(testloader, device, net)
