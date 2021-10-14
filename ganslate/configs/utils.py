import sys
import importlib
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ganslate.utils.io import import_attr

def init_config(conf, config_class):
    # Run-specific config
    conf = conf if isinstance(conf, DictConfig) else OmegaConf.load(str(conf))

    # Allows the framework to find user-defined, project-specific, classes and their configs
    if conf.project:

        assert isinstance(conf.project, str), "project needs to be a str path"

        # Import project as module with name "project"
        # https://stackoverflow.com/a/41595552
        project_path = Path(conf.project).resolve() / "__init__.py"
        assert project_path.is_file(), f"No `__init__.py` in project `{project_path}`."

        spec = importlib.util.spec_from_file_location("project", str(project_path))
        project_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(project_module)
        sys.modules["project"] = project_module

        logger.info(f"Project directory {conf.project} added to the"
                    " path as `project` to allow imports of modules from it.")


    # Make yaml mergeable by instantiating the dataclasses
    conf = instantiate_dataclasses_from_yaml(conf)
    # Merge default and run-specifig config
    return OmegaConf.merge(OmegaConf.structured(config_class), conf)


def instantiate_dataclasses_from_yaml(conf):
    """Goes through a config and instantiates the fields that are dataclasses.
    Each such dataclass should have an entry "_target_" which is used to import its dataclass
    class using that "_target_" + "Config" as class name.
    Instantiates the deepest dataclasses first as otherwise OmegaConf would throw an error.
    """
    for key in get_all_conf_keys(conf):
        # Get the field for that key
        field = OmegaConf.select(conf, key)
        if is_dataclass(field):
            dataclass = init_dataclass(field)
            # Update the field for that key with the newly instantiated dataclass
            OmegaConf.update(conf, key, OmegaConf.merge(dataclass, field), merge=False)
    return conf


def init_dataclass(field):
    """Initialize a dataclass. Requires the field to have a "_target_" entry.
    Assumes that the class name is of format "_target_" + "Config", e.g. "MRIDatasetConfig".
    """
    dataclass = f'{field["_target_"]}Config'
    dataclass = import_attr(dataclass)
    return OmegaConf.structured(dataclass)


def is_dataclass(field):
    """If a field contains `_target_` key, it is a dataclass."""
    return bool(isinstance(field, DictConfig) and "_target_" in field)


def get_all_conf_keys(conf):
    """Get all keys from a conf and order from them the deepest to the shallowest."""
    conf = OmegaConf.to_container(conf)
    keys = list(iterate_nested_dict_keys(conf))
    # Order deeper to shallower
    return keys[::-1]


def iterate_nested_dict_keys(dictionary):
    """Returns an iterator that returns all keys of a nested dictionary ordered
    from the shallowest to the deepest key. The nested keys are in the dot-list format,
    e.g. "gan.discriminator.in_channels".
    """
    if isinstance(dictionary, dict):
        current_level_keys = []
        for key in dictionary.keys():
            current_level_keys.append(key)
            yield key
        for key in current_level_keys:
            value = dictionary[key]
            for ret in iterate_nested_dict_keys(value):
                yield f"{key}.{ret}"
