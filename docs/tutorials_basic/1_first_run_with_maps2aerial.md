# Your First Run (Maps to Aerial Photo)

This tutorial will guide you on how to quickly get started with `ganslate` by doing running your first translation task. The task will be the translation of maps to aerial satellite photo. 


For both types of install (i.e. with `pip` and with `docker`), running the basic maps to aerial photo example is the same.

Once the installation is complete and you can access the [CLI as shown](using_cli.md), run
```console
ganslate your-first-run
```
On running this, a few options will show up that can be customized. You may also leave it at its default values. Once the prompts are completed, you will have a folder generated with a demo `maps` project in the path you specified. 

### Training
Next, you can run the training using the command below,

```console
ganslate train config=<path_specified>/default.yaml
```

!!! note
    If you have more than one GPU, then you can either run the training in distributed mode or [set CUDA_VISIBLE_DEVICES environment variable](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) to use only single GPUs.