from midaGAN.nn.utils import build_network_by_role

def build_G(conf, device):
    return build_network_by_role('generator', conf, device)
