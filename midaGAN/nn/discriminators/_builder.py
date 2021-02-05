from midaGAN.nn.utils import build_network_by_role


def build_D(conf, device):
    return build_network_by_role('discriminator', conf, device)
