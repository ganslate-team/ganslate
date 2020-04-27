import importlib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CustomDataLoader:
    '''
    Wraps an instance of Dataset class as either a regular Dataloader
    or as a DistributedSampler in case of distributed setup and 
    performs multi-threaded data loading.
    '''

    def __init__(self, conf):
        self.dataset = create_dataset(conf)
        self.sampler = None

        batch_size = conf.batch_size
        shuffle = conf.dataset.shuffle
        num_workers = conf.dataset.num_workers

        if conf.distributed:
            self.sampler = DistributedSampler(self.dataset, shuffle=shuffle)
            shuffle = False # no need to shufle Dataloader when DistributedSampler is shuffled
        
        self.dataloader = DataLoader(self.dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    sampler=self.sampler,
                                    num_workers=num_workers,
                                    pin_memory=True)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def create_dataset(conf):
    dataset = find_dataset_using_name(conf.dataset.mode)
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



