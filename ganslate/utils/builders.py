import copy
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import omegaconf

from ganslate.configs.config import Config
from ganslate.configs.utils import init_config
from ganslate.data.samplers import InfiniteSampler
from ganslate.nn.utils import init_net
from ganslate.utils import communication
from ganslate.utils.io import import_attr


def build_conf(omegaconf_args):
    cli = omegaconf.OmegaConf.from_dotlist(omegaconf_args)

    assert "config" in cli, "Please provide path to a YAML config using `config` option."
    yaml_conf = cli.pop("config")
    cli_conf_overrides = cli

    conf = init_config(yaml_conf, config_class=Config)
    return omegaconf.OmegaConf.merge(conf, cli_conf_overrides)


def build_loader(conf):
    """Builds the dataloader(s). If the config for dataset is a single dataset, it
    will return a dataloader for it, but if multiple datasets were specified,
    a list of dataloaders, one for each dataset, will be returned.
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
    dataset_class = import_attr(conf[conf.mode].dataset._target_)
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
    model_class = import_attr(conf.train.gan._target_)
    model = model_class(conf)
    return model


def build_G(conf, direction, device):
    assert direction in ['AB', 'BA']
    return build_network_by_role('generator', conf, direction, device)


def build_D(conf, domain, device):
    assert domain in ['B', 'A']
    return build_network_by_role('discriminator', conf, domain, device)


def build_network_by_role(role, conf, label, device):
    """Builds a discriminator or generator. TODO: document better 
    Parameters:
            role -- `generator` or `discriminator`
            conf -- conf
            label -- role-specific label 
            device -- torch device 
    """
    assert role in ['discriminator', 'generator']

    network_class = import_attr(conf.train.gan[role]._target_)

    network_args = dict(conf.train.gan[role])
    network_args.pop("_target_")
    network_args["norm_type"] = conf.train.gan.norm_type
    
    # Handle the network's channels settings
    if role == 'generator':
        in_out_channels = network_args.pop('in_out_channels')
        # TODO: This will enable support for both Dict and a single Tuple as 
        # mentioned in the config (configs/base.py#GeneratorInOutChannelsConfig) 
        # when OmegaConf will allow Union. Update comment when that happens.
        if isinstance(in_out_channels, omegaconf.dictconfig.DictConfig):
            in_out_channels = in_out_channels[label]
        network_args["in_channels"], network_args["out_channels"] = in_out_channels
    
    elif role == 'discriminator':
        # TODO: This will enable support for both Dict and a single Int as 
        # mentioned in the config (configs/base.py#DiscriminatorInChannelsConfig)
        # when OmegaConf will allow Union. Update comment when that happens.
        if isinstance(network_args["in_channels"] , omegaconf.dictconfig.DictConfig):
            network_args["in_channels"] = network_args["in_channels"][label]

    network = network_class(**network_args)
    return init_net(network, conf, device)
