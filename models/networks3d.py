import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import re


from models.generators.vnet import VNet, DeeperVNet
from models.discriminators.patchGAN_discriminator import NLayerDiscriminator
from models.losses.GAN_loss import GANLoss

###############################################################################
# Helper Functions
###############################################################################

# TODO: implement into VNet
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_device(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        net.to(device)
        

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_naive=False, 
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG.startswith('vnet_'):
        netG = VNet(num_classes=1, keep_input=use_naive)
    elif which_model_netG.startswith('deeper_vnet_'):
        netG = DeeperVNet(num_classes=1, keep_input=use_naive)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    init_weights(netG, init_type, gain=init_gain)
    init_device(netG, gpu_ids)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, 
             init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    init_weights(netD, init_type, gain=init_gain)
    init_device(netD, gpu_ids)
    return netD