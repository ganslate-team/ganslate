import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import ListConfig, OmegaConf

from midaGAN.configs.config import Config
from midaGAN.configs.utils import IMPORT_LOCATIONS, init_config
from midaGAN.data.samplers import InfiniteSampler
from midaGAN.nn.utils import init_net
from midaGAN.utils import communication
from midaGAN.utils.io import import_class_from_dirs_and_modules


def build_conf():
    cli = OmegaConf.from_cli()
    conf = init_config(cli.pop("config"), config_class=Config)
    return OmegaConf.merge(conf, cli)

def build_loader(conf):
    """Builds the dataloader(s). If the config for dataset is a single dataset, it
    will return a dataloader for it, but if multiple datasets were specified,
    a list of dataloaders, one for each dataset, will be returnet.
    """
    ############## Multi-dataset loaders #################
    if isinstance(conf[conf.mode].dataset, ListConfig):
        # Go through each dataset of the multi-dataset config, init
        # a dataloader for a dataset and add it to `loaders` list
        loaders = []
        for dataset_conf in conf[conf.mode].dataset:
            current_conf = copy.deepcopy(conf)
            # Set the config for the current dataset config
            current_conf[conf.mode].dataset = dataset_conf
            # Recursive call of this function to load single dataloaders
            loaders.append(build_loader(current_conf))
        return loaders

    ############## Single dataset loader #################
    name = conf[conf.mode].dataset.name
    dataset_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS["dataset"])
    dataset = dataset_class(conf)

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

    return DataLoader(dataset,
                      sampler=sampler,
                      batch_size=conf[conf.mode].batch_size,
                      num_workers=conf[conf.mode].dataset.num_workers,
                      pin_memory=conf[conf.mode].dataset.pin_memory)


def build_gan(conf):
    name = conf.train.gan.name
    model_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS["gan"])
    model = model_class(conf)
    return model


def build_G(conf, device):
    return build_network_by_role('generator', conf, device)


def build_D(conf, device):
    return build_network_by_role('discriminator', conf, device)


def build_network_by_role(role, conf, device):
    """Builds a discriminator or generator. TODO: document """
    assert role in ['discriminator', 'generator']

    name = conf.train.gan[role].name
    network_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS[role])

    network_args = dict(conf.train.gan[role])
    network_args.pop("name")
    network_args["norm_type"] = conf.train.gan.norm_type

    network = network_class(**network_args)
    return init_net(network, conf, device)
