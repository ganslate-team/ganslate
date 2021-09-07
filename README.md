# ganslate - simple and extensible GAN image-to-image translation framework.

For comprehensive documentation, visit: https://ganslate.netlify.app
Note: The documentation is still in progress! Suggestions and criticism, as well as contributions, are welcome! 

A [PyTorch](https://pytorch.org/) framework which aims to make GAN image-to-image translation more accessible to both beginner and advanced project with:

- Simple configuration system
- Extensibility for other datasets or architectures
- Documentation and [video walk-throughs (soon)](INSERT_YOUTUBE_PLAYLIST)

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
@misc{ganslate,
  author = {ganslate-team},
  doi = {},
  month = {9},
  title = {{ganslate - simple and extensible GAN image-to-image framework }},
  url = {https://github.com/ganslate-team/ganslate},
  year = {2020}
}
```