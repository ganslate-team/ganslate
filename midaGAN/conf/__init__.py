
import os
import sys
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from midaGAN.conf import config, datasets, gans, discriminators, 
from midaGAN.utils import import_class_from_dir


DIRS_WITH_CONFIGS = {
    "dataset": ["midaGAN.conf."]
}


def init_config(yaml_file):
    conf = OmegaConf.structured(config.Config)   # base config
    yaml_conf = OmegaConf.load(yaml_file)

    # Set environment variable for PROJECTS_DIR
    os.environ["PROJECTS_DIR"] = str(Path(yaml_file).resolve().parent)

    print(f"Projects Directory {os.environ.get('PROJECTS_DIR')} stored as environment variable.")
    OmegaConf.set_struct(conf, True)
    # Resolvers for more readable parameter initialization
    set_omegaconf_resolvers(yaml_conf)
    yaml_conf = instantiate_dataclasses_from_yaml(yaml_conf) # make yaml mergeable by instantiating the dataclasses
    return OmegaConf.merge(conf, yaml_conf)

def instantiate_dataclasses_from_yaml(conf):
    for key, entry in conf.items():
        if is_dataclass(entry, key):
            dataclass = init_dataclass(key, entry)
            OmegaConf.update(conf, key, OmegaConf.merge(dataclass, conf[key]))
    return conf

def init_dataclass(key, entry):
    print(f"Key: {key}")
    dataclass_name = f'{entry["name"]}Config'
    print(f"Dataclass: {dataclass_name}")
    # Check if dataclass is present in projects folder! 
    dataclass = import_class_from_dir(dataclass_name, [Path(os.environ.get("PROJECTS_DIR"))])
    
    # If not present in projects, use the conf folder to check datasets.
    if dataclass is None:
        module = getattr(sys.modules[__name__], key) # Rename modules and load them directly using getattr
        if hasattr(module, dataclass_name):
            dataclass = getattr(module, dataclass_name) 
        else:
            raise ValueError(f'YAML configuration incorrect with name:{entry["name"]} for {key}')
        
    dataclass = OmegaConf.structured(dataclass)
    return dataclass

def is_dataclass(entry, key):
    if isinstance(entry, DictConfig):
        if hasattr(sys.modules[__name__], key):
            return True
    return False
