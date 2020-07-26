from midaGAN.conf.base import config, datasets, gans, discriminators, generators
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import sys

MODULES = {
    'dataset': datasets,
    'gan': gans,
    'discriminator': discriminators,
    'generator': generators
}

def init_config(yaml_file):
    conf = OmegaConf.structured(config.Config)   # base config
    yaml_conf = OmegaConf.load(yaml_file)
    yaml_conf = instantiate_dataclasses_from_yaml(yaml_conf) # make yaml mergeable by instantiating the dataclasses
    return OmegaConf.merge(conf, yaml_conf)

def instantiate_dataclasses_from_yaml(conf):
    for key, entry in conf.items():
        if is_dataclass(entry):
            dataclass = init_dataclass(key, entry)
            dataclass = populate_dataclass_attrs(dataclass, conf[key])
            conf[key] = dataclass
    return conf

def populate_dataclass_attrs(dataclass, values_dict):
    check_if_contains_invalid_keys(values_dict, dataclass)
    for attribute in dataclass.keys():
        if attribute in values_dict.keys():
            if is_dataclass(values_dict[attribute]):
                values_dict = instantiate_dataclasses_from_yaml(values_dict)
            dataclass[attribute] = values_dict[attribute]
    return dataclass

def check_if_contains_invalid_keys(values_dict, dataclass):
    values_dict = {k:v for k,v in values_dict.items() if k != "class_name"}
    keys_dataclass = set(dataclass.keys())
    keys_to_be_set = set(values_dict.keys())
    invalid_keys = keys_to_be_set - keys_dataclass
    if len(invalid_keys) > 0:
        # TODO: you just see the key, not where it's nested, how to show the whole path?
        raise ValueError("YAML configuration contains following invalid key(s): {}.".format(invalid_keys))

def init_dataclass(key, entry):
    module = MODULES[key]
    dataclass = getattr(module, entry["class_name"])
    dataclass = OmegaConf.structured(dataclass)
    return dataclass

def is_dataclass(entry):
    if isinstance(entry, DictConfig):
        if "class_name" in entry.keys():
            return True
    return False
