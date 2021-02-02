import logging

from omegaconf import DictConfig, OmegaConf

from midaGAN import configs, data, nn, utils

logger = logging.getLogger(__name__)

IMPORT_LOCATIONS = {
    "dataset": [data],
    "gan": [nn.gans],
    "generator": [nn.generators],
    "discriminator": [nn.discriminators],
}


def init_config(conf, config_class):
    # Init default config
    base_conf = OmegaConf.structured(config_class)

    # Run-specific config
    if not isinstance(conf, DictConfig):
        conf = OmegaConf.load(str(conf))

    # Allows the framework to find user-defined, project-specific, dataset classes and their configs
    # TODO check if if else is needed
    project_dir = conf.train.project_dir if conf.train.project_dir else None 
    if project_dir:
        IMPORT_LOCATIONS["dataset"].append(project_dir)
        logger.info(f"Project directory {project_dir} added to the"
                    " path to allow imports of modules from it.")

    # Make yaml mergeable by instantiating the dataclasses
    conf = instantiate_dataclasses_from_yaml(conf)

    # Merge default and run-specifig config
    return OmegaConf.merge(base_conf, conf)


def instantiate_dataclasses_from_yaml(conf):
    """Goes through a config and instantiates the fields that are dataclasses.
    A field is a dataclass if its key can be found in the keys of the IMPORT_LOCATIONS.
    Each such dataclass should have an entry "name" which is used to import its dataclass
    class using that "name" + "Config" as class name.
    Instantiates the deepest dataclasses first as otherwise OmegaConf would throw an error.
    """
    for key in get_all_conf_keys(conf):
        # When dot-notation ('gan.discriminator'), use the last key as the name of it
        key_name = key.split('.')[-1]
        # Get the field for that key
        field = OmegaConf.select(conf, key)
        # See if that field is a dataclass itself by checking its name
        if is_dataclass(key_name, field):
            dataclass = init_dataclass(key_name, field)
            # Update the field for that key with the newly instantiated dataclass
            OmegaConf.update(conf, key, OmegaConf.merge(dataclass, field), merge=False)
    return conf


def init_dataclass(key, field):
    """Initialize a dataclass. Requires the field to have a "name" entry and a dataclass class
    whose destination can be found with IMPORT_LOCATIONS. Assumes that the class name is of
    format "name" + "Config", e.g. "MRIDatasetConfig".
    """
    dataclass_name = f'{field["name"]}Config'
    dataclass = utils.import_class_from_dirs_and_modules(dataclass_name, IMPORT_LOCATIONS[key])
    return OmegaConf.structured(dataclass)


def is_dataclass(key, field):
    """If a key is in the keys of IMPORT_LOCATIONS, it is a dataclass."""
    if isinstance(field, DictConfig):
        if key in IMPORT_LOCATIONS.keys():
            return True
    return False


def get_all_conf_keys(conf):
    """Get all keys from a conf and order from them the deepest to the shallowest."""
    conf = OmegaConf.to_container(conf)
    keys = list(utils.iterate_nested_dict_keys(conf))
    # Order deeper to shallower
    return keys[::-1]
