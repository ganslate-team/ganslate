from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler

from ganslate.nn import separable


def init_net(network, conf, device):
    init_weights(network, conf.train.gan.weight_init_type, conf.train.gan.weight_init_gain)
    return network.to(device)


def init_weights(net, weight_init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            if weight_init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif weight_init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif weight_init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif weight_init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    f"initialization method `{weight_init_type}` is not implemented")
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def get_conv_layer_3d(is_separable=False):
    if is_separable:
        return separable.SeparableConv3d
    else:
        return nn.Conv3d


def get_conv_transpose_layer_3d(is_separable=False):
    if is_separable:
        return separable.SeparableConvTranspose3d
    else:
        return nn.ConvTranspose3d


def get_norm_layer_2d(norm_type='instance'):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError(f"Normalization layer `{norm_type}` not supported")


def get_norm_layer_3d(norm_type='instance'):
    if norm_type == 'batch':
        return nn.BatchNorm3d
    elif norm_type == 'instance':
        return nn.InstanceNorm3d
    else:
        raise NotImplementedError(f"Normalization layer `{norm_type}` not supported")


def is_bias_before_norm(norm_type='instance'):
    """When using BatchNorm, the preceding Conv layer does not use bias,
    but it does if using InstanceNorm.
    """
    if norm_type == 'instance':
        return True
    elif norm_type == 'batch':
        return False
    else:
        raise NotImplementedError(f"Normalization layer `{norm_type}` not supported")


def get_scheduler(optimizer, conf):
    """Return a scheduler that keeps the same learning rate for the first <conf.train.n_iters> epochs
    and linearly decays the rate to zero over the next <conf.train.n_iters_decay> epochs.
    Parameters:
        optimizer -- the optimizer of the network
        TODO
    """

    def lambda_rule(iter_idx):
        start_iter = 1
        if conf.train.checkpointing.load_iter:
            start_iter += conf.train.checkpointing.load_iter
        lr_l = 1.0 - max(
            0, iter_idx + start_iter - conf.train.n_iters) / float(conf.train.n_iters_decay + 1)
        return lr_l

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def get_network_device(network):
    """Returns the device of the network. Assumes that the whole network is on a single device."""
    return next(network.parameters()).device
