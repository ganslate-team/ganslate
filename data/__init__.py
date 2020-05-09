import importlib
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.distributed import DistributedSampler
from util.sampler import InfiniteSampler


def build_dataloader(conf):
    batch_size = conf.batch_size
    shuffle = conf.dataset.shuffle
    num_workers = conf.dataset.num_workers

    dataset = build_dataset(conf)
    sampler = InfiniteSampler(size=len(dataset), shuffle=shuffle, seed=None) # TODO: seed to conf and other components? Remember that torch.Generator advises to use high values for seed, if the defines int is small should we multiply it by some factor?
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader

def build_dataset(conf):
    dataset = find_dataset_using_name(conf.dataset.name)
    return dataset(conf)

def find_dataset_using_name(dataset_name):
    # TODO: nicer
    # Given the option --dataset_mode [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of Dataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, Dataset):
            dataset = cls
            
    if dataset is None:
        print("In %s.py, there should be a subclass of torch.utils.data.Dataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
        exit(0)

    return dataset



