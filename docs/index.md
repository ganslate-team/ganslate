[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5494572.svg)](https://doi.org/10.5281/zenodo.5494572)

# ganslate

A [PyTorch](https://pytorch.org/) framework which aims to make GAN image-to-image translation more accessible to both beginner and advanced project with:

- Simple configuration system
- Extensibility for other datasets or architectures
- Documentation and [video walk-throughs](INSERT_YOUTUBE_PLAYLIST)

## Features

- 2D and 3D support
- Mixed precision
- Distributed training
- Tensorboard and [Weights&Biases](https://wandb.ai/site) logging
- Natural and medical image support
- A range of generator and discriminator architectures

## Available GANs

- Pix2Pix ([paper](https://www.google.com/search?q=pix2pix+paper&oq=pix2pix+paper&aqs=chrome.0.0l2j0i22i30l2j0i10i22i30.3304j0j7&sourceid=chrome&ie=UTF-8))
- CycleGAN ([paper](https://arxiv.org/abs/1703.10593))
- RevGAN ([paper](https://arxiv.org/abs/1902.02729))
- CUT (Contrastive Unpaired Translation) ([paper](https://arxiv.org/abs/2007.15651))

## Projects
`ganslate` was used in:

- Project 1
- Project 2

## Citation

If you used `ganslate` in your project, please cite:

```text
@software{ibrahim_hadzic_2021_5494572,
  author       = {Ibrahim Hadzic and
                  Suraj Pai and
                  Chinmay Rao and
                  Jonas Teuwen},
  title        = {ganslate-team/ganslate: Initial public release},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.5494572},
  url          = {https://doi.org/10.5281/zenodo.5494572}
}
```