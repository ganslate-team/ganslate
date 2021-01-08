import importlib
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from midaGAN import configs
from midaGAN.utils import communication, import_class_from_dirs_and_modules
from midaGAN.data.samplers import InfiniteSampler

logger = logging.getLogger(__name__)


def build_loader(conf):
    dataset = build_dataset(conf)

    if conf.is_train:
        sampler = InfiniteSampler(size=len(dataset), shuffle=conf.dataset.shuffle)
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
                        batch_size=conf.batch_size,
                        sampler=sampler,
                        num_workers=conf.dataset.num_workers,
                        pin_memory=True)
    return loader


def build_dataset(conf):
    name = conf.dataset.name
    import_locations = configs.utils.initializers.IMPORT_LOCATIONS
    dataset_class = import_class_from_dirs_and_modules(name, import_locations["dataset"])
    dataset = dataset_class(conf)
    return dataset
