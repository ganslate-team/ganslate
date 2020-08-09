

def build_D(conf, device):
    name = conf.discriminator.model
    norm_type = conf.gan.norm_type

    discriminator_args = dict(conf.discriminator)
    discriminator_args.pop("model") # used only to select the model

    if name == 'patchgan2d':
        discriminator = PatchGAN2D(**discriminator_args, norm_type=norm_type)
    elif name == 'patchgan':
        discriminator = PatchGAN(**discriminator_args, norm_type=norm_type)
    else:
        raise NotImplementedError(' model name [%s] is not recognized' % name_D)

    init_type = conf.gan.weight_init_type
    init_gain = conf.gan.weight_init_gain
    init_weights(discriminator, init_type, init_gain)

    return discriminator.to(device)