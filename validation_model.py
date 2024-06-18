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

import sys

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

# import matplotlib.pyplot as plt
# import numpy as np

# functions to show an image

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


def my_acc_calculator(val_dataloader, model, distance, epoch = 199, val_dataset = "test"):
                
    actual_label = []
    predicted_label = []
    pos_threshold = 0.75
    # target_label = ['negative pair', 'positive pair']
    for i, data in enumerate(val_dataloader, 0):
        inputs, labels = data[0].cuda(), data[1].cuda()
        
        # print labels
        # inputs, labels = data[0], data[1]
        # imshow(torchvision.utils.make_grid(inputs))
        # print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
        
        outputs = model(inputs)
        mat = distance(outputs)
        for idx in range(mat.size(dim=1)):
            vec = mat[idx]
            actual = [1 if x == labels[idx] else 0 for x in labels]
            predicted = [1 if (y - pos_threshold) > 1e-9 else 0 for y in vec]
            
            actual_label += actual
            predicted_label += predicted
        
    total = len(actual_label)
    # print(actual_label)
    # print(predicted_label)
    correct = 0
    for i in range(total):
        correct += actual_label[i] == predicted_label[i]
    accuracy = 100.0 * (float)(correct) / (float)(total) 
    output_str = f"Test set accuracy (pos/neg pair)  on dataset {val_dataset} = {accuracy}"
    print(output_str)
    # at epoch {epoch + 1}
    # print(output_str, file=fv)
    # ts_writer.add_scalar(f"accuracy per {val_freq} epoch on dataset {val_dataset}", accuracy, epoch + 1)


    
# load data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# val_weight = sys.argv[0]
# val_weight.seek(0)
val_weight = "./DDP_epoch50_batch16_rank0.pth"
# print(val_weight)
# print(torch.load(val_weight))
net = Net()
net.load_state_dict(torch.load(val_weight))
net.eval()
net.cuda()
distance = distances.CosineSimilarity()

my_acc_calculator(testloader, net, distance)