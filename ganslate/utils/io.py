import json
from pathlib import Path


def mkdirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def make_dataset_of_files(root, extensions):
    """The root of dataset contains files of the given extension."""
    root = Path(root).resolve()
    assert root.is_dir(), f"{root} is not a valid directory"
    paths = [root / file for file in root.iterdir() if has_extension(file, extensions)]
    return sorted(paths)


def make_recursive_dataset_of_files(root, extensions):
    root = Path(root).resolve()
    assert root.is_dir(), f"{root} is not a valid directory"
    paths = []
    for ext in extensions:
        paths.extend(list(root.rglob(f"*{ext}")))
    return sorted(paths)


def has_extension(file, extensions):
    suffix = Path(file).suffixes
    suffix = "".join(suffix)  # join necessary to get ".nii.gz" and similar suffixes properly
    return any(ext in suffix for ext in extensions)


def make_dataset_of_directories(root, extensions):
    """The root of dataset contains folders for each data point. Each data point folder has to have
    (at least) one file of the specified extension. The dataset has to define which file it takes from
    such folder. Useful when using a dataset that stores, for example, an image and a mask together
    in their own folder.
    """
    root = Path(root).resolve()
    assert root.is_dir(), f"{root} is not a valid directory"
    paths = [root / folder for folder in root.iterdir() if (root / folder).is_dir()]
    paths = [folder for folder in paths if has_files_with_extension(folder, extensions)]
    return sorted(paths)


def make_recursive_dataset_of_directories(root, extensions):
    files = make_recursive_dataset_of_files(root, extensions)
    directories = list({f.parent for f in files})
    return directories


def has_files_with_extension(folder, extensions):
    for ext in extensions:
        if not ext.startswith("."):
            ext = "." + ext
        files_in_folder = list(folder.glob(f"*{ext}"))
        if files_in_folder:
            return True
    return False


def find_paths_containing_pattern(path, pattern, recursive=False):
    path = Path(path)
    paths = path.rglob(pattern) if recursive else path.glob(pattern)
    return list(paths)


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def import_attr(module_attr):
    module, attr = module_attr.rsplit(".", 1)
    module = __import__(module, fromlist=[attr])
    return getattr(module, attr)

# TODO: All the code below was taken from MONAI source code since this particular code
# is not available yet in MONAI available through pip. Once is is, remove this code and
# use it from MONAI. https://docs.monai.io/en/latest/data.html#monai.data.utils.decollate_batch

from typing import Optional, List, Any
import collections
import torch


def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if isinstance(obj, torch.Tensor):
        return int(obj.dim()) > 0  # a 0-d tensor is not iterable
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def decollate(data: dict, batch_size: Optional[int] = None) -> List[dict]:
    """De-collate a batch of data (for example, as produced by a `DataLoader`).

    Returns a list of dictionaries. Each dictionary will only contain the data for a given batch.

    Images originally stored as (B,C,H,W,[D]) will be returned as (C,H,W,[D]). Other information,
    such as metadata, may have been stored in a list (or a list inside nested dictionaries). In
    this case we return the element of the list corresponding to the batch idx.

    Return types aren't guaranteed to be the same as the original, since numpy arrays will have been
    converted to torch.Tensor, and tuples/lists may have been converted to lists of tensors

    For example:

    .. code-block:: python

        batch_data = {
            "image": torch.rand((2,1,10,10)),
            "image_meta_dict": {"scl_slope": torch.Tensor([0.0, 0.0])}
        }
        out = decollate_batch(batch_data)
        print(len(out))
        >>> 2

        print(out[0])
        >>> {'image': tensor([[[4.3549e-01...43e-01]]]), 'image_meta_dict': {'scl_slope': 0.0}}

    Args:
        data: data to be de-collated.
        batch_size: number of batches in data. If `None` is passed, try to figure out batch size.
    """
    if not isinstance(data, dict):
        raise RuntimeError(
            "Only currently implemented for dictionary data (might be trivial to adapt).")
    if batch_size is None:
        for v in data.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break
    if batch_size is None:
        raise RuntimeError("Couldn't determine batch size, please specify as argument.")

    def torch_to_single(d: torch.Tensor):
        """If input is a torch.Tensor with only 1 element, return just the element."""
        return d if d.numel() > 1 else d.item()

    def decollate(data: Any, idx: int):
        """Recursively de-collate."""
        if isinstance(data, dict):
            return {k: decollate(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            out = data[idx]
            return torch_to_single(out)
        elif isinstance(data, list):
            if len(data) == 0:
                return data
            if isinstance(data[0], torch.Tensor):
                return [torch_to_single(d[idx]) for d in data]
            if issequenceiterable(data[0]):
                return [decollate(d, idx) for d in data]
            return data[idx]
        raise TypeError(f"Not sure how to de-collate type: {type(data)}")

    return [{key: decollate(data[key], idx) for key in data.keys()} for idx in range(batch_size)]
