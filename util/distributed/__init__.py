import os
import torch
import util.distributed.multi_gpu
import logging

logger = logging.getLogger(__name__)

def init_distributed(local_rank):
    num_gpu = int(os.environ.get('WORLD_SIZE', 1))
    if num_gpu > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        multi_gpu.synchronize()
        logger.info(f'Number of GPUs available in world: {num_gpu}.')
    else:
        raise ValueError("Distributed ON but but running single process") # TODO make nicer