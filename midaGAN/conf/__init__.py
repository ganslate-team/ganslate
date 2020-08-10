
import os
import sys
import logging

from pathlib import Path
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from midaGAN.conf import config
from midaGAN.utils import import_class_from_dirs_and_modules

from midaGAN.nn import (gans, generators, discriminators)
from midaGAN import datasets

logger = logging.getLogger(__name__)

# But it's not used only for config, it's also used for importing models, datasets etc (do a search to see) TODO: change name?
CONFIG_LOCATIONS = {
    "dataset": [datasets],
    "gan": [gans],
    "generator": [generators],
    "discriminator": [discriminators],
}

def init_config(yaml_file):
    # Allows the framework to find user-defined, project-specific, dataset classes and their configs
    current_project_folder = Path(yaml_file).resolve().parent.parent
    CONFIG_LOCATIONS["dataset"].append(current_project_folder)
    logger.info(f"Project directory {current_project_folder} added to path to allow imports of modules from it.")

    # Init default config
    conf = OmegaConf.structured(config.Config)
    OmegaConf.set_struct(conf, True)

    # Load run-specific config 
    yaml_conf = OmegaConf.load(yaml_file)
    set_omegaconf_resolvers(yaml_conf) 
    yaml_conf = instantiate_dataclasses_from_yaml(yaml_conf) # make yaml mergeable by instantiating the dataclasses
    
    # Merge default and run-specifig config
    return OmegaConf.merge(conf, yaml_conf)

def instantiate_dataclasses_from_yaml(conf):
    for key, entry in conf.items():
        if is_dataclass(entry, key):
            dataclass = init_dataclass(key, entry)
            OmegaConf.update(conf, key, OmegaConf.merge(dataclass, conf[key]))
    return conf

def init_dataclass(key, entry):
    dataclass_name = f'{entry["name"]}Config'
    dataclass = import_class_from_dirs_and_modules(dataclass_name, CONFIG_LOCATIONS[key])
    return OmegaConf.structured(dataclass)

def is_dataclass(entry, key):
    if isinstance(entry, DictConfig):
        if key in CONFIG_LOCATIONS.keys():
            return True
    return False

def set_omegaconf_resolvers(conf):
    # Infer length of an object with interpolations using omegaconf
    # Here till issue closed: https://github.com/omry/omegaconf/issues/100
    try:
        OmegaConf.register_resolver(
        "len",
        lambda x: len(
            conf.select(x),
        ))   

    # Added exception handler for profiling with torch bottleneck
    except AssertionError:
        logger.info('Already registered resolver')