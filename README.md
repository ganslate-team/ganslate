

## Installation
For all dependencies use conda env 
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