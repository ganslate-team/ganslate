import importlib
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import midaGAN
from midaGAN.data.samplers import InfiniteSampler
from midaGAN.utils import import_class_from_dirs_and_modules, communication

logger = logging.getLogger(__name__)

def build_loader(conf):
    dataset = build_dataset(conf)

    if conf.gan.is_train:
        sampler = InfiniteSampler(size=len(dataset), 
                                  shuffle=conf.dataset.shuffle, 
                                  seed=None) # TODO: seed to conf and other components? Remember that torch.Generator advises to use high values for seed, if the defines int is small should we multiply it by some factor?
    else:
        sampler = None
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset,
                                         shuffle=False,
                                         num_replicas=communication.get_world_size(),
                                         rank=communication.get_local_rank()) # TODO: verify that this indeed should be local rank and not rank

    loader =  DataLoader(dataset,
                         batch_size=conf.batch_size,
                         sampler=sampler,
                         num_workers=conf.dataset.num_workers,
                         pin_memory=True)
    return loader

def build_dataset(conf):
    name = conf.dataset.name
    import_locations = midaGAN.conf.IMPORT_LOCATIONS
    dataset_class = import_class_from_dirs_and_modules(name, import_locations["dataset"])
    dataset = dataset_class(conf)
    return dataset