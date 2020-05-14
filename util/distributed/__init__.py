import os
import torch
import util.distributed.multi_gpu

def init_distributed():
    num_gpu = int(os.environ.get('WORLD_SIZE', 1))
    if num_gpu > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        multi_gpu.synchronize()
        print(f'Number of GPUs available in world: {num_gpu}.')
    else:
        raise ValueError("Distributed ON but but running single process") # TODO make nicer