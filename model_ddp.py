import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms

import time
import datetime
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from torch.profiler import profile, record_function, ProfilerActivity


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

def ddp_example(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # b = B // world_size

    model = Net().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as p:
        with record_function("model_inference"):
            with torch.cuda.device(rank):
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                batch_size = 256
                epoch = 10

                trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                        download=False, transform=transform)

                sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=False, num_workers=0, sampler=sampler)

                # dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

                
                # tik = time.time()
                startTime = datetime.datetime.now()
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                for ep in range(epoch):
                    trainloader.sampler.set_epoch(epoch)     
                    for (i, data) in enumerate(trainloader, 0):
                        # forward pass
                        x, label = data[0].to(rank), data[1].to(rank)
                        outputs = ddp_model(x)
                        # backward pass
                        loss_func(outputs, label).backward()
                        # update parameters
                        optimizer.step()
                    nowTime = datetime.datetime.now()
                    output_str = f"epoch # {ep + 1} done, {nowTime}"
                    print(output_str)
                # end.record()
                # print(f"DDP rank-{rank} execution time (ms) by CUDA event {start.elapsed_time(end)}")
                # torch.cuda.synchronize()
                # tok = time.time()
                nowTime = datetime.datetime.now()
                print(f"DDP rank-{rank} execution time (s) by Python time {nowTime - startTime} ")
                cleanup()
                if (rank == 0):
                    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=100))

def cleanup():
    dist.destroy_process_group()

def main():

    world_size = 2
    # with profile(
    # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as p:
        # with record_function("model_inference"):
    mp.spawn(ddp_example,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    # print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__=="__main__":
    main()
