import os
import json
import glob

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dataset_of_files(root, extensions=['.npy']):
    """The root of dataset contains files of the given extension."""
    assert os.path.isdir(root), '%s is not a valid directory' % root
    paths = [os.path.join(root, file) for file in os.listdir(root) if has_extension(file, extensions)]
    return sorted(paths)
    
def has_extension(file, extensions):
    return any(file.endswith(ext) for ext in extensions)

def make_dataset_of_folders(root, extensions=['.npy']):
    """The root of dataset contains folders for each data point. Each data point folder has to have
    (at least) one file of the specified extension. The dataset has to define which file it takes from
    such folder. Useful when using a dataset that stores, for example, an image and a mask together
    in their own folder.
    """
    assert os.path.isdir(root), '%s is not a valid directory' % root
    paths = [os.path.join(root, folder) for folder in os.listdir(root) \
             if os.path.isdir(os.path.join(root, folder))]
    paths = [folder for folder in paths if has_files_with_extension(folder, extensions)]
    return sorted(paths)

def has_files_with_extension(folder, extensions):
    for ext in extensions:
        files_in_folder = list(glob.glob(os.path.join(folder, "*"+ext)))
        if files_in_folder:
            return True
    return False


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)