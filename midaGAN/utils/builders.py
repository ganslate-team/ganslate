import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf

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
    if "multi_dataset" in conf[conf.mode] and conf[conf.mode].multi_dataset is not None:
        assert conf[conf.mode].dataset is None, "Use either `dataset` or `multi_dataset`."

        # Go through each dataset of the multi-dataset config,
        # initialize it, and add to the `loaders` dict
        loaders = {}
        for dataset_name, dataset_conf in conf[conf.mode].multi_dataset.items():
            # Avoids affecting the original conf
            current_conf = copy.deepcopy(conf)
            # Set the config for the current dataset config
            current_conf[conf.mode].dataset = dataset_conf
            # Allows instantiation of the dataset as otherwise the initial assertion fails
            current_conf[conf.mode].multi_dataset = None
            # Initialize the single dataloaders and assigning it to its name in dict
            loaders[dataset_name] = build_loader(current_conf)

        return loaders

    ############## Single dataset loader #################
    name = conf[conf.mode].dataset.name
    dataset_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS)
    dataset = dataset_class(conf)

    # Prevent DDP from running on a dataset smaller than the total batch size over processes
    if torch.distributed.is_initialized():
        ddp_batch_size = communication.get_world_size() * conf[conf.mode].dataset.batch_size
        if ddp_batch_size > len(dataset):
            raise RuntimeError(f"Dataset has {len(dataset)} examples, while the effective "
                               f"batch size equals to {ddp_batch_size}. Distributed mode does "
                               f"not work as expected in this situation.")

    if conf.mode == "train":
        sampler = InfiniteSampler(size=len(dataset), shuffle=True)
    else:
        sampler = None
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset,
                                         shuffle=False,
                                         num_replicas=communication.get_world_size(),
                                         rank=communication.get_rank())
    return DataLoader(dataset,
                      sampler=sampler,
                      batch_size=conf[conf.mode].batch_size,
                      num_workers=conf[conf.mode].dataset.num_workers,
                      pin_memory=conf[conf.mode].dataset.pin_memory)


def build_gan(conf):
    name = conf.train.gan.name
    model_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS)
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
    network_class = import_class_from_dirs_and_modules(name, IMPORT_LOCATIONS)

    network_args = dict(conf.train.gan[role])
    network_args.pop("name")
    network_args["norm_type"] = conf.train.gan.norm_type

    network = network_class(**network_args)
    return init_net(network, conf, device)
