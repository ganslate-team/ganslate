# Loading Your Own Data into ganslate with Custom Pytorch Datasets

`ganslate` can be run on your own data through creating [your own Pytorch Dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). 


## Integrating your Pytorch Dataset for Training
Once you have your custom Dataset, it needs to be modified as `ganslate` expects certain structure to make the Dataset class compatible with the framework. 

Namely, it expects *atleast* the following,

1. Specific return format from the `__getitem__` function
2. Dataclass configuration for your dataset

#### Specific return format from the `__getitem__` function
A sample of the format expected to be returned:
```python
def __getitem__(self, index):
    ...

    return {'A': ..., 'B': ...}
```
A dictionary with the following keys is expected to be returned

1. `A`  - corresponds to `torch.Tensor` image (2D/3D) from domain A
2. `B`  - corresponds to `torch.Tensor` image (2D/3D) from domain B


#### Dataclass configuration for your dataset
The Dataset can be dynamically configured through Dataclasses as a part of the [OmegaConf configuration system](https://github.com/omry/omegaconf). Apart from configuration, the Dataclass is also important to allow the framework to easily import your Dataset while training. A sample of this can be found in the [default ImageDataset](https://github.com/Maastro-CDS-Imaging-Group/midaGAN/blob/26564fa721f71c024aa88fb278ecba7de748e55c/midaGAN/data/image_dataset.py#L15) provided with `ganslate`. 

The structure of the Dataclass configuration 
```python
from dataclasses import dataclass
from ganslate import configs

@dataclass
class YourDatasetNameConfig(configs.base.BaseDatasetConfig): # Your dataset always needs to inherit the BaseDatasetConfig
    # Define additional parameters below, these parameters are passed to
    # the dataset and can be used for dynamic configuration. 
    # Examples of parameters
    flip: bool = True

```

`YourDatasetName` is to be consistent with the name of your Pytorch Dataset. The name is also used to import the Dataset module. 


To allow your Dataset to access parameters defined in the Dataclass configuration, the `___init___` function of the Dataset can be modified.
```python
from torch.utils.data import Dataset

class YourDatasetName(Dataset):
    def __init__(self, conf): # `conf` contains the entire configuration yaml 
        self.flip = conf[conf.mode].dataset.flip # Accessing the flip parameter defined in the dataclass

```

#### Importing your Dataset with `ganslate`
Your Dataset along with its Dataclass configuration can be placed in the `projects` folder under a specific project name.
For example,
```
projects/
    your_project/
        your_dataset.py # Contains both the Dataset and Dataclass configuration
        default_docker.yaml
```

Modify the `default_docker.yaml`
```yaml
project: "./projects/your_project" # This needs to point to the directory where your_dataset.py is located
train:
    dataset:
        _target_: "YourDatasetName"
        root: "<path_to_datadir>" # Path to where the data is 
        # Additional parameters
        flip: True
```

Apart from this, make sure the other parameters in the `default_docker.yaml` are set appropriately. [Refer to configuring your training with yaml files](configuration.md). 

You can now run training with your custom Dataset! Run this command from the root of the repository,
```python
python tools/train.py config=projects/your_project/default_docker.yaml
```







