import importlib
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from midaGAN.configs.utils import IMPORT_LOCATIONS
from midaGAN.utils import communication
from midaGAN.utils.io import import_class_from_dirs_and_modules
from midaGAN.data.samplers import InfiniteSampler

logger = logging.getLogger(__name__)


def build_loader(conf):
    dataset = build_dataset(conf)

    if conf.mode == "train":
        sampler = InfiniteSampler(size=len(dataset), shuffle=conf.train.dataset.shuffle)
    else:
        sampler = None
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(
                dataset,
                shuffle=False,
                num_replicas=communication.get_world_size(),
                # TODO: should it be rank instead?
                rank=communication.get_local_rank())

    loader = DataLoader(dataset,
                        sampler=sampler,
                        batch_size=conf[conf.mode].batch_size,
                        num_workers=conf[conf.mode].dataset.num_workers,
                        pin_memory=conf[conf.mode].dataset.pin_memory)
    return loader


def build_dataset(conf):
    name = conf[conf.mode].dataset.name
    dataset_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS["dataset"])
    dataset = dataset_class(conf)
    return dataset
