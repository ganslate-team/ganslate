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



# ------------ Reduce and All Reduce --------------

def reduce(input_data, average=False, all_reduce=False):
    """
    Reduce any type of data [int, float, tensor, dict, list, tuple] by summing or averaging
    the value(s) using one of the methods:
    (1) rank 0 reduce (torch.distributed.reduce)  - communicates the sum or average of 
                                                    all processes to the process of rank 0 only
    (2) all reduce (torch.distributed.all_reduce) - communicates the sum or average of 
                                                    all processes to each process
    
    Parameters:
        input_dict (int, float, tensor, dict, list, tuple) -- data for reduction
        average (bool) -- true if the results should be averaged
        all_reduce (bool) -- true if communicating the reduced value to all processes, 
                             otherwise process of rank 0 only

    Returns the sum or average of the input data from across processes. 
    """
    if get_world_size() < 2:
        return input_data

    # use device that is compatible with the backend (e.g. nccl does not support CPU tensors)
    device = torch.device('cuda' if torch.distributed.get_backend() == 'nccl' else 'cpu')

    with torch.no_grad():
        if isinstance(input_data, torch.Tensor):
            reduced_data = reduce_tensor(input_data, average, all_reduce, device)
        
        elif isinstance(input_data, dict):
            reduced_data = reduce_dict(input_data, average, all_reduce, device)
        
        elif isinstance(input_data, list) or isinstance(input_data, tuple):
            reduced_data = reduce_list_tuple(input_data, average, all_reduce, device)
        
        elif isinstance(input_data, int) or isinstance(input_data, float):
            reduced_data = reduce_int_float(input_data, average, all_reduce, device)
        
        else:
            raise NotImplementedError("Reduction on data type %s is not implemented." % str(type(data)))
    return reduced_data


is_not_tensor = lambda x: not isinstance(x, torch.Tensor)
is_float_or_int = lambda x: isinstance(x, float) or isinstance(x, int)


def reduce_tensor(tensor, average, all_reduce, device):
    """Reduce a tensor"""
    tensor = tensor.clone().to(device)
    if all_reduce:
        torch.distributed.all_reduce(tensor)
    else:
        torch.distributed.reduce(tensor, dst=0)

    if average and (get_rank() == 0 or all_reduce):
        tensor /= get_world_size()
    return tensor


def reduce_int_float(input_value, average, all_reduce, device):
    """Reduce an integer or float by converting it to tensor, 
    performing reduction and, finally, casting it back to the initial type. 
    """
    data_type = type(input_value)  # save the original data type
    tensor = torch.Tensor([input_value])  # convert to tensor
    tensor = reduce_tensor(tensor, average, all_reduce, device)
    reduced_value = data_type(tensor.item())  # cast back to int or float
    return reduced_value


def reduce_dict(input_dict, average, all_reduce, device):
    """ Reduce a dict by extracting all of its values into a tensor and communicating it.    . 
    Returns a dict with the same fields as input_dict, after reduction. If its values were 
    int or float, they are converted to tensors.

    * if order of the keys matters, convert the dict to OrderedDict before passing it to the function.
    """
    names = []
    values = []
    for k in input_dict.keys():
        names.append(k)
        value = input_dict[k]
        # if float or integer and not tensor, convert to tensor
        if is_not_tensor(value):
            if is_float_or_int(value): 
                value = torch.Tensor([value])  
            else:
                raise NotImplementedError("Dictionary reduction supported only if \
                                           its values are tensors, floats or integers.")
        values.append(value)

    values = torch.stack(values, dim=0).to(device) # convert the list of tensors to a single tensor 

    if all_reduce:
        torch.distributed.all_reduce(values)
    else:
        torch.distributed.reduce(values, dst=0)

    if average and (get_rank() == 0 or all_reduce):
        values /= get_world_size()  

    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_list_tuple(input_data, average, all_reduce, device):
    """ Reduce a list or tuple whose elements are either tensors, floats or integers. 
    Returns reduced list/tuple with its elements as tensors.
    """
    data_type = type(input_data)  # save the original data type
    # convert tuple/list values to tensors if they are floats or integers
    for i in range(len(input_data)):
        value = input_data[i]
        if is_not_tensor(value):
            if is_float_or_int(value): 
                input_data[i] = torch.Tensor([value])
            else:
                raise NotImplementedError("List/tuple reduction supported only if \
                                            its values are tensors, floats or integers.")
    values = torch.stack(input_data, dim=0).to(device) # convert list/tuple of tensors to a single tensor
    
    if all_reduce:
        torch.distributed.all_reduce(values)
    else:
        torch.distributed.reduce(values, dst=0)

    if average and (get_rank() == 0 or all_reduce):
        values /= get_world_size()

    reduced_sequence = data_type(values)  # cast it back to tuple or list
    return reduced_sequence
