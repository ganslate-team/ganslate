# Datasets

If you are familiar with the [standard data loading workflow in _PyTorch_](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), `ganslate` uses similar dataset classes which are derived from `torch.utils.data.Dataset` to define the data fetching and preprocessing pipeline. Additionally, `ganslate`'s datasets have an associated Python `dataclass` in which all the data-related settings are defined.



--------------------------------------------------------------
## The `PairedImageDataset` and `UnpairedImageDataset` classes

Two classes - `PairedImageDataset` and `UnpairedImageDataset` - are supplied by the `ganslate` by default which can be used out-of-the-box on your image data. 

The `PairedImageDataset` class enables loading from the storage an _A_-_B_ image pair given a common index and applying optional joint transformations on the pair. This class is to be used in paired training as well during validation and/or testing when paired data is available.

On the other hand, `UnpairedImageDataset` fetches randomly a domain _A_ image and a domain _B_ image and applies optional transformations on each independently. As the name suggects, this class is meant to be used for unpaired training.

### Input and Output
Both classes expect your data directory to be structured in the following manner
```text
<train_dataset_root_dir>
    |
    |- A
    |   |- ...
    |   |- ...
    |
    |- B
    |   |- ...
    |   |- ...
```

And if using validation or testing data
```text
<val_dataset_root_dir>
    |
    |- A
    |   |- ...
    |   |- ...
    |
    |- B
    |   |- ...
    |   |- ...
```

where the sub-directories _A_ and _B_ contain the images. In situations where paired data is provided (i.e. paired training or all valdation/testing), the ordering of images in _A_ corresponds to the ordering of images in _B_, meaning that the first _A_ image the first _B_ image are pairs and so on. Images with extensions `.jpg`, `.jpeg`, and `png` are supported.


Both image dataset classes implemet a `__getitem__` method that outputs a sample dictionary of the following form
```python
sample = {'A': a_tensor, 'B': b_tensor}
```
where the each tensor is of shape (`C`, `H`, `W`).


### Available Settings
The configuration `dataclasses` associated with both default image datasets are inherited from `configs.base.BaseDatasetConfig` which two settings common to all dataset classes. These are `num_workers` and `pin_memory` which are the settings for the `torch.utils.data.DataLoader` used by `ganslate` internally. 


The two image datasets have additional settings which are same across the two datasets. These are:

- `image_channels`: Refers to the number of channels in the images. Only the images with 1 and 3 channels are supported (i.e. grayscale and RGB), and the channels should be the same across _A_ and _B_ images (i.e. either both should be grayscale or both should be RGB).

- `preprocess`: Accepts a tuple of predefined strings that defines the preprocessing instructions. These predefined strings include `resize`, `scale_width`, `random_zoom`, `random_crop`, and `random_flip` of which `resize` and `scale_width` specify the initial resizing operations (choose either one or none), whereas the rest correspond to the random transforms used as data augmentation during training. An example value for the `preprocess` settings is `('resize', 'random_crop', 'random_flip')`

Note: In `PairedImageDataset`, these transforms are applied jointly to the _A_ and _B_ images, whereas in `UnpairedImageDataset`, they are applied on each image independently.

- `load_size`: This parameter accepts a tuple that specifies the size (`H`, `W`) to which the images are to be loaded from the storage and resized as a result of the `resize` preprocessing instruction.

- `final_size`: This parameter accepts a tuple that specifies the size (`H`, `W`) to which the images are converted as a result of the random transforms. This is the final size of the images that should be expected from the dataloader.

Note: When not using random transforms (for example, during validation), specify the `final_size` the same as `load_size`.



-------------------------
## Custom Dataset Classes

It is also possible to define your custom dataset class for use-cases requiring specialized processing for the data, for exmaple, in case of medical images. See [Your First Project](../tutorials_basic/2_new_project.md) for more more details on creating custom datasets.