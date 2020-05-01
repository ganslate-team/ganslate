import functools
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm3d, affine=True) # TODO: check sync version https://pytorch.org/docs/stable/nn.html#torch.nn.SyncBatchNorm
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        return None
    else:
        raise NotImplementedError('Normalization layer [%s] not supported' % norm_type)

def is_bias_before_norm(norm_type='instance'):
    """When using BatchNorm, the preceding Conv layer does not use bias, 
    but it does if using InstanceNorm.
    """
    if norm_type == 'instance' or norm_type == 'none':
        return True
    elif norm_type == 'batch':
        return False
    else:
        raise NotImplementedError('Normalization layer [%s] not supported' % norm_typeorm_type)

def get_scheduler(optimizer, conf):
    """Return a scheduler that keeps the same learning rate for the first <conf.n_epochs> epochs
    and linearly decays the rate to zero over the next <conf.n_epochs_decay> epochs.
    Parameters:
        optimizer          -- the optimizer of the network
        TODO
    """
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + conf.continue_epoch - conf.n_epochs) / float(conf.n_epochs_decay + 1)
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