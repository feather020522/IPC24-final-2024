import torch
import torchvision
import torchvision.transforms as transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
# import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
# print('why')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # torch.distributed.init_process_group(backend="nccl", rank=rank, init_method="env://", world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

# training
def training(rank, world_size, trainloader, testloader):    
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
    
    # loss function
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    # accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    # DDP
    epochs = 50
    setup(rank, world_size)
    my_net = Net().to(rank)
    ddp_net = DDP(my_net, device_ids=[rank])
    optimizer = optim.Adam(ddp_net.parameters(), lr=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data[0].to(rank), data[1].to(rank)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            
            # parallelization
            outputs = ddp_net(inputs)
        
            # dataparallel
            # netParallel = torch.nn.DataParallel(net, device_ids=[0,1])
            # outputs = netParallel(inputs)
            
            # outputs = net(inputs)
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
            #     running_loss = 0.0

        
        # get current time
        nowTime = datetime.datetime.now()
        output_str = f"epoch # {epoch + 1} done, {nowTime}, rank {rank}"
        print(output_str)
        # print(output_str, file=fs)

    print(f'{startTime - nowTime}')
    cleanup()
    my_acc_calculator(testloader, ddp_net, distance)
    
    PATH = f'./DDP_epoch50_batch16_rank{rank}.pth'
    torch.save(ddp_net.state_dict(), PATH)

    


if __name__ == '__main__':
    # load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    # parameter

    # epochs = 50
    # net = Net()
    # net.cuda()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    # the time start training
    nowTime = datetime.datetime.now()
    output_str = f"Start training, {nowTime}"
    print(output_str)

    # training


    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    mp.spawn(training,
            args=(world_size,trainloader,testloader,),
            nprocs=world_size,
            join=True)

# python -m torch.distributed.launch xxx.py