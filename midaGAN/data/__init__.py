import importlib
import logging
from torch.utils.data import Dataset, DataLoader
from midaGAN.utils.sampler import InfiniteSampler
from midaGAN.utils import str_to_class

logger = logging.getLogger(__name__)

def build_loader(conf):
    batch_size = conf.batch_size
    shuffle = conf.dataset.shuffle
    num_workers = conf.dataset.num_workers

    dataset = build_dataset(conf)
    sampler = InfiniteSampler(size=len(dataset), shuffle=shuffle, seed=None) # TODO: seed to conf and other components? Remember that torch.Generator advises to use high values for seed, if the defines int is small should we multiply it by some factor?
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader

def build_dataset(conf):
    name = conf.dataset.name
    dataset_class = str_to_class(f"midaGAN.data.{name.replace('Dataset', '_dataset').lower()}", name)
    dataset = dataset_class(conf)
    return dataset