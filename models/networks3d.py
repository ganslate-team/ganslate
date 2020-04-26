import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import re

from models.generators.vnet import VNet, DeeperVNet
from models.discriminators.patchGAN_discriminator import NLayerDiscriminator

# TODO: implement into VNet
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True) # TODO: check sync version https://pytorch.org/docs/stable/nn.html#torch.nn.SyncBatchNorm
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, n_epochs, n_epochs_decay, epoch_count):
    """Return a scheduler that keeps the same learning rate for the first <opt.n_epochs> epochs
    and linearly decays the rate to zero over the next <opt.n_epochs_decay> epochs.
    Parameters:
        optimizer          -- the optimizer of the network
        TODO
    """
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def init_weights(net, weight_init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if weight_init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif weight_init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif weight_init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif weight_init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % weight_init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
        

def define_G(n_channels_input, n_channels_output, n_first_filters_G, model_G, norm_layer='batch', use_memory_saving=True, 
             weight_init_type='normal', weight_init_gain=0.02, device=torch.device('cuda:0')):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm_layer)
    keep_input = not use_memory_saving

    if model_G.startswith('vnet_'):
        netG = VNet(num_classes=1, keep_input=keep_input)
    elif model_G.startswith('deeper_vnet_'):
        netG = DeeperVNet(num_classes=1, keep_input=keep_input)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % model_G)

    init_weights(netG, weight_init_type, gain=weight_init_gain)
    netG.to(device)
    return netG


def define_D(n_channels_input, n_first_filters_D, model_D, n_layers_D=3, norm_layer='batch', use_sigmoid=False, 
             weight_init_type='normal', weight_init_gain=0.02, device=torch.device('cuda:0')):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm_layer)

    if model_D == 'basic':
        netD = NLayerDiscriminator(n_channels_input, n_first_filters_D, n_layers=2, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif model_D == 'n_layers':
        netD = NLayerDiscriminator(n_channels_input, n_first_filters_D, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif model_D == 'pixel':
        netD = PixelDiscriminator(n_channels_input, n_first_filters_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  model_D)
    init_weights(netD, weight_init_type, gain=weight_init_gain)
    netD.to(device)
    return netD