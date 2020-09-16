
import os
import sys
import logging

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import midaGAN
from midaGAN.conf import config
from midaGAN.utils import import_class_from_dirs_and_modules


logger = logging.getLogger(__name__)

# But it's not used only for config, it's also used for importing models, datasets etc (do a search to see) TODO: change name?
IMPORT_LOCATIONS = {
    "dataset": [midaGAN.data],
    "gan": [midaGAN.nn.gans],
    "generator": [midaGAN.nn.generators],
    "discriminator": [midaGAN.nn.discriminators],
}

def init_config(conf, config_class=config.Config, contains_dataclasses=True):
    # Init default config
    base_conf = OmegaConf.structured(config_class)

    # Run-specific config 
    if not isinstance(conf, DictConfig):
        conf = OmegaConf.load(str(conf))
    set_omegaconf_resolvers(conf)

    # Allows the framework to find user-defined, project-specific, dataset classes and their configs
    if conf.project_dir:
        IMPORT_LOCATIONS["dataset"].append(conf.project_dir)
        logger.info(f"Project directory {conf.project_dir} added to path to allow imports of modules from it.")

    # Make yaml mergeable by instantiating the dataclasses
    if contains_dataclasses:
        conf = instantiate_dataclasses_from_yaml(conf) 
    
    # Merge default and run-specifig config
    return OmegaConf.merge(base_conf, conf)

def instantiate_dataclasses_from_yaml(conf):
    for key, entry in conf.items():
        if is_dataclass(entry, key):
            dataclass = init_dataclass(key, entry)
            OmegaConf.update(conf, key, OmegaConf.merge(dataclass, conf[key]), merge=False)
    return conf

def init_dataclass(key, entry):
    dataclass_name = f'{entry["name"]}Config'
    dataclass = import_class_from_dirs_and_modules(dataclass_name, IMPORT_LOCATIONS[key])
    return OmegaConf.structured(dataclass)

def is_dataclass(entry, key):
    if isinstance(entry, DictConfig):
        if key in IMPORT_LOCATIONS.keys():
            return True
    return False

def set_omegaconf_resolvers(conf):
    # Infer length of an object with interpolations using omegaconf
    # Here till issue closed: https://github.com/omry/omegaconf/issues/100
    try:
        OmegaConf.register_resolver("len", lambda x: len(conf.select(x)))
    # Added exception handler for profiling with torch bottleneck
    except AssertionError:
        logger.info('Already registered resolver')