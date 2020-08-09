
import importlib
# import pathlib

# def list_modules_from_directory(dir_path):
#     return [str(module_name.stem) for module_name in pathlib.Path('.').glob('*.py')]

def str_to_class(module_name, class_name):
    """
    Convert a string to a class
    From: https://stackoverflow.com/a/1176180/576363

    Parameters
    ----------
    module_name : str
        e.g. direct.data.transforms
    class_name : str
        e.g. Identity
    Returns
    -------
    object
    """
    try:
        # Load the module, will raise ModuleNotFoundError if module cannot be loaded.
        module = importlib.import_module(module_name)
        # Get the class, will raise AttributeError if class cannot be found.
        the_class = getattr(module, class_name)
        return the_class
        
    except AttributeError:
        return None