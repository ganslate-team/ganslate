
import importlib


def str_to_class(module_name: str, class_name: str) -> Callable:
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

    # Load the module, will raise ModuleNotFoundError if module cannot be loaded.
    module = importlib.import_module(module_name)
    # Get the class, will raise AttributeError if class cannot be found.
    the_class = getattr(module, class_name)

    return the_class