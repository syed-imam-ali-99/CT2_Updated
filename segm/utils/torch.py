import os
import torch
import torch.distributed


"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode, local_rank=0):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1
    if "RANK" in os.environ:
        dist_rank = int(os.environ["RANK"])
    else:
        dist_rank = 0
    gpu_id = local_rank
    distributed = world_size > 1
    use_gpu = bool(mode and torch.cuda.is_available())
    if use_gpu:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True
    else:
        if mode:
            print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")
