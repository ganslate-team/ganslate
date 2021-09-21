# Intro to Configuration

`ganslate`'s configuration system is based on [OmegaConf](https://github.com/omry/omegaconf). If you have used [hydra](https://hydra.cc/docs/intro/), you might already be familiar with it. Otherwise, please refer to [OmegaConf documentation](https://omegaconf.readthedocs.io/)

## Configuring a Run
Setting up an experiment is done through a YAML file. This is an example of a run on `cityscapes` dataset:
```
# Project path from which custom, non-framework, can be loaded.
# If not using any custom code, set it to `null`
project: "./projects/cityscapes_label2photo/"

train:
    output_dir: "./checkpoints/label2photo_cyclegan/"
    cuda: True
    n_iters: 297500  # 2975 images x 100 epochs 
    n_iters_decay: 297500
    batch_size: 1
    mixed_precision: False

    logging:
        freq: 100
        wandb:
            project: "cityscapes_label2photo"
            run: "cyclegan_trial"
        
    checkpointing:
        freq: 10000

    dataset: 
        _target_: project.datasets.Label2PhotoDataset
        root: "~/Downloads/Datasets/Cityscapes_label2photo/train"
        load_size: [572, 286]
        crop_size: [512, 256]
        paired: False  # Unpaired, for training
        random_flip: True
        random_crop: True
        masking: False
        num_workers: 8

    gan:  
        _target_: ganslate.nn.gans.unpaired.CycleGAN

        generator:  
            _target_: ganslate.nn.generators.Resnet2D
            n_residual_blocks: 9
            in_channels: 3
            out_channels: 3

        discriminator:  
            _target_: ganslate.nn.discriminators.PatchGAN2D
            n_layers: 3
            in_channels: 3

        optimizer:
            lr_D: 0.0002
            lr_G: 0.0002
            lambda_AB: 10.0
            lambda_BA: 10.0
            lambda_identity: 0
            proportion_ssim: 0
    
    metrics:
        discriminator_evolution: True
        ssim: True


val:
    freq: 200

    dataset: 
        _target_: project.datasets.Label2PhotoDataset
        root: "~/Downloads/Datasets/Cityscapes_label2photo/val"
        load_size: [512, 256]
        paired: True  # Paired, to compute similarity metrics 
        random_flip: False
        random_crop: False
        masking: False
        num_workers: 8

    metrics:
        cycle_metrics: False

# test:
    # Not defined

# infer:
    # Not defined
```

### Config structure overview
At the root level, the configuration is separated into `train`, `val`, `test`, and `inference`. Each of these phases has a separate configuration, but some options in `val`, `test`, and `inference` default to their equivalents in `train`. For example, since `gan` isn't specified in `val`, it is interpolated (copied) from `train`, using [OmegaConf's interpolation](https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html#interpolations).
<br />You don't need to define all the phases at once - running `ganslate train` uses `train` and, if defined, `val` config. With `ganslate test` it will only need the `test` config, and with `ganslate infer` the `infer` config.

At the sub-root levels, the configurations are separated into logical units such as `checkpointing`, `logging`, `gan`, etc. When an option's default value works for the experiment, you can ommit it in YAML. For example, `train.checkpointing` has the `tensorboard` option set to `False` by default. As a result, you don't have to explicitly set it to `False` when you don't want to log with tensorboard.

!!! note 
    Description of all configuration options can be found [here](MISSING LINK). <!--- TODO: Update the link -->

!!! note 
    `ganslate` logs the whole config, including the default and overriden values, at the start of each run. 

### Overriding config from the command line
If you need a quick override of an option, you can do so from the command line instead of modifying the YAML file. For example, specifying another path to the training dataset from the command line would look like this:
```
ganslate train config=<CONFIG_PATH> train.dataset.root=./Downloads/cityscapes/train/
```

## Configuration Structure Definition

The *configuration structure* in `ganslate` is defined using [Python dataclasses and OmegaConf](https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html). Defining config like that enables modular config design, so that options differ based on the architecture, dataset, or some other choice. Additionally, it enables static type checking.

The following is `CycleGAN`'s config structure:

```
from dataclasses import dataclass
from ganslate import configs

@dataclass
class CycleGANConfig(configs.base.BaseGANConfig):
    """CycleGAN Config"""
    pool_size: int = 50
    optimizer: OptimizerConfig = OptimizerConfig
```
`CycleGANConfig` specifies two different options: `pool_size`, and `optimizer`. Each has the type specified as well as the default value. For instance, `pool_size` option needs to be an `int` and defaults to `50`. If we tried to set `pool_size` to a string `"fifty"`, OmegaConf would raise an error.

We can also see that `CycleGANConfig` inherits from `configs.base.BaseGANConfig`. `BaseGANConfig` is an abstraction of a GAN config, and defines several options necessary for any GAN.

```
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class BaseGANConfig:
    """Base GAN config."""
    _target_: str = MISSING  # refers to the class that this config describes
    norm_type: str = "instance"
    weight_init_type: str = "normal"
    weight_init_gain: float = 0.02

    optimizer: BaseOptimizerConfig = MISSING
    generator: BaseGeneratorConfig = MISSING
    # Discriminator optional as it is not used in inference
    discriminator: Optional[BaseDiscriminatorConfig] = None
```

As a result of inheriting from it, `CycleGANConfig` has the options defined in `BaseGANConfig`, such as `norm_type` or `generator`.

!!! note
    When writing custom architecture or dataset classes, you also need to write such dataclass configs for them, as demonstrated in [Custom Architecture](custom_architecture.md) and [Custom Dataset](custom_dataset.md) sections. 