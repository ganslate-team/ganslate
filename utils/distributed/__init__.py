import os
import torch
from utils.distributed import communication

def init_distributed():
    num_gpu = int(os.environ.get('WORLD_SIZE', 1))
    if num_gpu > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        communication.synchronize()
        print(f'Number of GPUs available in world: {num_gpu}.')
    else:
        raise ValueError("Distributed ON but but running single process") # TODO make nicer