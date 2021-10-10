# Model Architectures and Loss Functions



--------------------
## GAN Architectures

`ganslate` provides implementations of several popular image translation GANs, which you can use out-of-the-box in your projects. Here is the list of currently supported GAN architectures:

1. Pix2Pix
    - Class: `ganslate.nn.gans.paired.Pix2PixConditionalGAN`
    - Data requirements: Paired pixel-wise aligned domain _A_ and domain _B_ images
    - Original paper: Isola et. al - Image-to-Image Translation with Conditional Adversarial Networks ([arXiv](https://arxiv.org/abs/1611.07004))

2. CycleGAN
    - Class: `ganslate.nn.gans.unpaired.CycleGAN` 
    - Data requirements: Unpaired domain _A_ and domain _B_ images
    - Original paper: Zhu et. al - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks ([arXiv](https://arxiv.org/abs/1703.10593))

3. RevGAN
    - Class: `ganslate.nn.gans.unpaired.RevGAN`
    - Data requirements: Unpaired domain _A_ and domain _B_ images
    - Original paper: Ouderaa et. al - Reversible GANs for Memory-efficient Image-to-Image Translation ([arXiv](https://arxiv.org/abs/1902.02729))

4. CUT
    - Class: `ganslate.nn.gans.unpaired.CUT`
    - Data requirements: Unpaired domain _A_ and domain _B_ images
    - Original paper: Park et. al - Contrastive Learning for Unpaired Image-to-Image Translation ([arXiv](https://arxiv.org/abs/2007.15651))

`ganslate` defines an abstract base class `ganslate.nn.gans.base.BaseGAN` ([source](https://github.com/ganslate-team/ganslate/nn/gans/base.py)) that implements some of the basic functionalty common to all the aforementioned GAN architectures, such as methods related to model setup, saving, loading, learning rate update, etc. Additionally, it also declares certain abstract methods whose implementation might differ across various GAN architectures, such as the forward pass and backpropagation logic. Each of the aforementioned GAN architectures inherits from `BaseGAN` and implements the necessary abstract methods.

The `BaseGAN` class has an associated `dataclass` at `ganslate.configs.base.BaseGANConfig` that defines all its basic settings including the settings for optimizer, generator, and discriminator. Since the different GAN architectures have their own specific settings, each of them also has an associated configuration `dataclass` that inherits from `ganslate.configs.base.BaseGANConfig` and defines additional architecture-specific settings.

As a result to its extensible design, `ganslate` additionally enables users to modify the existing GANs by overriding certain functionalities or to define their own custom image translation GAN from scratch. The former is discussed in the context of loss functions as part of the basic tutorial [Your First Project](../tutorials_basic/2_new_project.md). Whereas, the latter is part of the advanced tutorial [Writing Your Own GAN Class from Scratch](../tutorials_advanced/1_custom_gan_architecture.md).



--------------------------------------------
## Generator and Discriminator Architectures

Generators and discriminators are defined in `ganslate` as regular _PyTorch_ modules derived from `torch.nn.Module`. 

Following is the list of the available generator architectures:

1. ResNet variants (Original ResNet paper - [arXiv](https://arxiv.org/abs/1512.03385)):
    - 2D ResNet: `ganslate.nn.generators.Resnet2D`
    - 3D ResNet: `ganslate.nn.generators.Resnet3D`
    - Partially-invertible ResNet generator: `ganslate.nn.generators.Piresnet3D`

2. U-Net variants (Original U-Net paper - [arXiv](https://arxiv.org/abs/1505.04597)):
    - 2D U-Net: `ganslate.nn.generators.Unet2D`
    - 3D U-Net: `ganslate.nn.generators.Unet3D`

3. V-Net variants (Original V-Net paper - [arXiv](https://arxiv.org/abs/1606.04797))
    - 2D V-Net: `ganslate.nn.generators.Vnet2D`
    - 3D V-Net: `ganslate.nn.generators.Vnet3D`
    - Partially-invertible 3D V-Net generator with Self-Attention: `ganslate.nn.generators.SelfAttentionVnet3D`


And here is the list of the available discriminator architectures:

1. PatchGAN discriminator variants (PatchGAN originally described in the Pix2Pix paper - [arXiv](https://arxiv.org/abs/1611.07004))
    - 2D PatchGAN: `ganslate.nn.discriminators.PatchGAN2D`
    - 3D PatchGAN: `ganslate.nn.discriminators.PatchGAN3D`
    - Multiscale 3D PatchGAN: `ganslate.nn.discriminators.MultiScalePatchGAN3D`
    - 3D PatchGAN with Self-Attention: `ganslate.nn.discriminators.SelfAttentionPatchGAN3D`



-----------------
## Loss Functions

Several different loss function classes are provided in the `ganslate` package. These include different flavors of the adversarial loss as well as various GAN architecture-specific losses.

1. Adversarial loss
    - Class: `ganslate.nn.losses.adversarial_loss.AdversarialLoss`
    - Variants: `'vanilla'` (original adversarial loss based on cross-entropy), `'lsgan'` (least-squares loss), `'wgangp'` (Wasserstein-1 distance with gradient penalty), and `'nonsaturating'`.

2. Pix2Pix losses
    - Class: `ganslate.nn.losses.pix2pix_losses.Pix2PixLoss`
    - Components: 
        - Pixel-to-pixel L1 loss between synthetic image and ground truth (weighted by the scalar `lambda_pix2pix`).

3. CycleGAN losses
    - Class: `ganslate.nn.losses.cyclegan_losses.CycleGANLosses`
    - Components: 
        - Cycle-consistency loss based on L1 distance (_A-B-A_ and _B-A-B_ components separated weighted by `lambda_AB` and `lambda_BA`, respectively). Option to compute cycle-consistency as using a weighted sum of L1 and SSIM losses (weights defined by the hyperparameter `proportion_ssim`).
        - Identity loss implemented with L1 distance.

3. CUT losses
    - Class: `ganslate.nn.losses.cut_losses.PatchNCELoss`
    - Components:
        - PatchNCE loss