![YAPF Formatting Check](https://github.com/Maastro-CDS-Imaging-Group/midaGAN/workflows/YAPF%20Formatting%20Check/badge.svg)


## Installation
To install apex run the 
```
bash install_apex.sh
```

For all other dependencies use conda env 
```
conda env update -f environment.yml
```



## Running the code

Run training:
```
python tools/train.py config="<PATH_TO_YAML>" 
```

Override options from yaml config, example - change batch_size to 4:
```
python tools/train.py config="<PATH_TO_YAML>" batch_size=4
```

Run in distributed mode, single node example:
```
python -m torch.distributed.launch --use_env --nproc_per_node <NUM_GPUS_PER_NODE> tools/train.py config="<PATH_TO_YAML>"
```
Find more about other options by running `python -m torch.distributed.launch --help`


## Thesis Code and Modules

The structure of code developed solely for the purpose of this thesis is found below,
```
- modules/  # This folder contains implementations for adaptive strategies and  structure-consistency losses investigated as a part of the thesis.
  - losses/ # Contains implementation of specific loss functions used in the thesis
- projects/
  - suraj_msc_thesis_expts/ # Contains pytorch datasets and experiment yaml files for experiments defined on the CBCT-CT dataset. 
    - Frequency_Representation_Notebook.ipynb # Notebook that allows exploratory analysis to determine if structure loss would be beneficial
  - aerial_to_maps/ # Contains pytorch datasets and experiment yaml files for experiments defined on the maps-> aerial photo dataset. 
```
Note that multiple other features were implemented as supporting elements to the thesis, that are found in the main package. 


