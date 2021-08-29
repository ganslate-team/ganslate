# Your First Run (horse2zebra)

For both types of install, running the basic horse2zebra example is similar.
#### Data download
The horse2zebra dataset can be downloaded using instructions below,
1. Open a terminal inside the cloned repository and run, 

```console
cd projects/horse2zebra
bash download_cyclegan_dataset.sh horse2zebra .
```

!!! note
    The horse2zebra dataset will be downloaded at the root-level of the ganslate directory. This can be changed by providing a <data_dir> to the command `bash download_cyclegan_dataset.sh horse2zebra <data_dir>. However, the yaml files need to be manually changed. For non-advanced users, it is best to stick to the default location

### Training
Next, you can run the training using the command below,

```console
cd /code
python tools/train.py config=projects/horse2zebra/experiments/default.yaml
```

!!! note
    If you have more than one GPU, then you can either run the training in distributed mode or [set CUDA_VISIBLE_DEVICES environment variable](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) to use only single GPUs.