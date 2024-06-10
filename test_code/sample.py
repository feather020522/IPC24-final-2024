import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

import time

X = 100
B = 200

def ddp_example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    b = B // world_size
    # create local model
    model = nn.Linear(X, X).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    with torch.cuda.device(rank):
        tik = time.time()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            # forward pass
            outputs = ddp_model(torch.randn(b, X).to(rank))
            labels = torch.randn(b, X).to(rank)
            # backward pass
            loss_fn(outputs, labels).backward()
            # update parameters
            optimizer.step()
        end.record()
        print(f"DDP rank-{rank} execution time (ms) by CUDA event {start.elapsed_time(end)}")
        torch.cuda.synchronize()
        tok = time.time()
        print(f"DDP rank-{rank} execution time (s) by Python time {tok - tik} ")


def dp_example():
    b = B  # don't need to divide by 2 here as DataParallel will scatter inputs
    model = nn.Linear(X, X).to(0)
    # construct DDP model
    dp_model = DP(model, device_ids=[0, 1])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(dp_model.parameters(), lr=0.001)

    tik = time.time()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        # forward pass
        outputs = dp_model(torch.randn(b, X).to(0))
        labels = torch.randn(b, X).to(0)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
    end.record()
    print(f"DP execution time (ms) by CUDA event: {start.elapsed_time(end)}")
    torch.cuda.synchronize()
    tok = time.time()
    print(f"DP execution time (s) by Python time: {tok - tik} ")


def main():
    dp_example()

    world_size = 2
    mp.spawn(ddp_example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    main()
