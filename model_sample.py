import torch
import torchvision
import torchvision.transforms as transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
# import numpy as np

# load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# model definition

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d: (input_channels, output_channels, kernel_size, stride)
        # 3 * h (?) * w (?)
        self.conv1 = nn.Conv2d(3, 27, 3, 1)
        # 27 * 254 * 126
        self.pool1 = nn.MaxPool2d(2, 2)
        # 27 * 127 * 63
        self.conv2 = nn.Conv2d(27, 81, 9, 3)
        # 81 * 40 * 19
        self.pool2 = nn.MaxPool2d(2, 2)
        # 81 * 20 * 9
        self.conv3 = nn.Conv2d(81, 162, 1, 2)
        # 162 * 9 * 4
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(162, 81)
        self.fc3 = nn.Linear(81, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# loss function
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
# accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

# parameter

epochs = 200
net = Net()
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# training
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data[0].cuda(), data[1].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # loss.item()
        # if i % 100 == 99:
        #     running_loss /= (i % 100 + 1)
        #     output_str = f'[{epoch + 1}] [data:{i+1}] loss: {running_loss:.6f}'
        #     print(output_str)
        #     # losses_list.append(running_loss)
        #     # ts_writer.add_scalar(f"Loss per {loss_freq} mini_batch", running_loss, epoch * len(room_dataloader_train) + (i + 1))
        #     running_loss = 0.0
            
        # if i % loss_freq == loss_freq - 1 or i == len(trainloader) - 1:
        #     running_loss /= (i % loss_freq + 1)
        #     output_str = f'[{epoch + 1}] [data:{i+1}] loss: {running_loss:.6f}'
        #     print(output_str, file=ft)
        #     losses_list.append(running_loss)
        #     ts_writer.add_scalar(f"Loss per {loss_freq} mini_batch", running_loss, epoch * len(room_dataloader_train) + (i + 1))
        #     running_loss = 0.0

        # print("pass")

    # get current time
    nowTime = datetime.datetime.now()
    output_str = f"epoch # {epoch + 1} done, {nowTime}"
    print(output_str)
    # print(output_str, file=fs)

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)