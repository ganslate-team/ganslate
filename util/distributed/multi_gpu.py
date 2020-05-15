# coding=utf-8
"""
Copyright (c) DIRECT Contributors
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Taken from Detectron 2, licensed under Apache 2.0.
# https://github.com/facebookresearch/detectron2/blob/989f52d67d05445ccd030d8f13d6cc53e297fb91/detectron2/utils/comm.py
# Changes:
# - Docstring to match the rest of the library.
# - Calls to other subroutines which do not exist in DIRECT.
# - Extra logging.


import torch
import logging
import numpy as np
import pickle
import functools

from typing import List

logger = logging.getLogger(__name__)


def synchronize():
    """
    Synchronize processes between GPUs. Wait until all devices are available.
    Function returns nothing in a non-distributed setting too.
    """
    if not torch.distributed.is_available():
        logger.info('torch.distributed: not available.')
        return

    if not torch.distributed.is_initialized():
        logger.info('torch.distributed: not initialized.')
        return

    if torch.distributed.get_world_size() == 1:
        logger.info('torch distributed: world size is 1')
        return

    torch.distributed.barrier()


def get_rank() -> int:
    """
    Get rank of the process, even when torch.distributed is not initialized.
    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0

    return torch.distributed.get_rank()


def is_main_process() -> bool:
    """
    Simple wrapper around get_rank().
    Returns
    -------
    bool
    """
    return get_rank() == 0


def get_world_size() -> int:
    """
    Get number of compute device in the world, returns 1 in case multi device is not initialized.
    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def shared_random_seed() -> int:
    """
    All workers must call this function, otherwise it will deadlock.
    Returns
    -------
    A random number that is the same across all workers. If workers need a shared RNG, they can use this shared seed to
    create one.
    """

    seed = torch.randint(2**31, (1,)).cuda() # TODO verify this works
    if is_main_process():
        if get_world_size() > 1:
            torch.distributed.broadcast(seed, 0) # TODO doesnt seem alright
    print('infinite sampler seed:', seed.item())
    return seed.item()
