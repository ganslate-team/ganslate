
import os
import sys
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from midaGAN.conf import config, datasets, gans, discriminators, 
from midaGAN.utils import str_to_class

def set_env_project_folder(yaml_file):
    os.environ["PROJECTS_DIR"] = str(Path(yaml_file).resolve().parent)
    print(f"Projects Directory {os.environ.get('PROJECTS_DIR')} stored as environment variable.")

def init_config(yaml_file):
    conf = OmegaConf.structured(config.Config)   # base config
    yaml_conf = OmegaConf.load(yaml_file)
    yaml_conf = instantiate_dataclasses_from_yaml(yaml_conf) # make yaml mergeable by instantiating the dataclasses
    return OmegaConf.merge(conf, yaml_conf)

def instantiate_dataclasses_from_yaml(conf):
    for key, entry in conf.items():
        if is_dataclass(entry):
            dataclass = init_dataclass(key, entry["name"])
            OmegaConf.update(conf, key, OmegaConf.merge(dataclass, conf[key]))
    return conf

def init_dataclass(key, name):
    #module = MODULES[key]
    #dataclass = getattr(module, entry["name"] + "Config")
    if key == "dataset":
        module_name = name.replace('Dataset', '_dataset').lower()
        class_name = name + "Config"
        dataclass = str_to_class(f"midaGAN.data.{module_name}", class_name)
        if dataclass is None:
            dataclass = str_to_class(f"{}")

    dataclass = OmegaConf.structured(dataclass)
    return dataclass

def is_dataclass(entry):
    if isinstance(entry, DictConfig):
        if "name" in entry.keys():
            return True
    return False
