import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
import memcnn
import re
from apex.parallel import DistributedDataParallel

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

    #print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], is_distributed=False):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #print(gpu_ids)
        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        net.to(device)
        if is_distributed: 
            net = DistributedDataParallel(net)
        # use DataParallel only if it's not distributed and there are multiple GPUs
        elif len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_naive=False, init_type='normal', init_gain=0.02, gpu_ids=[], is_distributed=False, n_downsampling=2):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    # if which_model_netG.startswith('srcnn_'):
    #     depth = int(re.findall(r'\d+', which_model_netG)[0])
    #     netG = SrcnnGenerator3d(input_nc, output_nc, depth, ngf)
    # elif which_model_netG.startswith('edsrF_'):
    #     depth = int(re.findall(r'\d+', which_model_netG)[0])
    #     netG = EdsrFGenerator3d(input_nc, output_nc, depth, ngf, use_naive)
    # el
    if which_model_netG.startswith('vnet_'):
        #depth = int(re.findall(r'\d+', which_model_netG)[0])
        #netG = EdsrFGenerator3d(input_nc, output_nc, depth, ngf, use_naive)
        netG = VNet(num_classes=1, keep_input=use_naive)
    elif which_model_netG.startswith('deeper_vnet_'):
        netG = DeeperVNet(num_classes=1, keep_input=use_naive)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids, is_distributed)


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', 
             use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[], is_distributed=False):
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
    return init_net(netD, init_type, init_gain, gpu_ids, is_distributed)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

        

# ---------------- V-NET -------------------

class VnetRevBlock(nn.Module):
    def __init__(self, nchan, keep_input):
        super(VnetRevBlock, self).__init__()
        
        invertible_module = memcnn.AdditiveCoupling(
            Fm=self.build_conv_block(nchan//2),
            Gm=self.build_conv_block(nchan//2)
        )
        # TODO: keep_input option
        self.rev_block = memcnn.InvertibleModuleWrapper(fn=invertible_module, 
                                                        keep_input=keep_input, 
                                                        keep_input_inverse=keep_input)

    def build_conv_block(self, nchan):
        block = nn.Sequential(nn.Conv3d(nchan, nchan, kernel_size=5, padding=2),
                              nn.BatchNorm3d(nchan),
                              nn.PReLU(nchan))
        return block

    def forward(self, x, inverse=False):
        if inverse:
            return self.rev_block.inverse(x)
        else:
            return self.rev_block(x)


class InputTransition(nn.Module):
    def __init__(self, outChans):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.PReLU(16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels (or is it right to say duplicate the input for 16 times to operate "torch.add()"?? By Chao) 
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1) # changed dim = 0 to 1 to operate on channels, and have "x16" the same size as "out" to operate "torch.add()". By Chao.
        out = self.relu(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, keep_input):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv_ab = self.build_down_conv(inChans, outChans)
        self.down_conv_ba = self.build_down_conv(inChans, outChans)
        self.core = nn.Sequential(*[VnetRevBlock(outChans, keep_input) for _ in range(nConvs)])
        self.relu = nn.PReLU(outChans)

    def build_down_conv(self, inChans, outChans):
        return nn.Sequential(nn.Conv3d(inChans, outChans, kernel_size=2, stride=2),
                             nn.BatchNorm3d(outChans),
                             nn.PReLU(outChans))

    def forward(self, x, inverse=False):
        if inverse:
            down_conv = self.down_conv_ba
            core = reversed(self.core)
        else:
            down_conv = self.down_conv_ab
            core = self.core

        down = down_conv(x)
        out = down
        for i, block in enumerate(core):
            if i == 0:
                #https://github.com/silvandeleemput/memcnn/issues/39#issuecomment-599199122
                if inverse:
                    block.rev_block.keep_input_inverse = True
                else:
                    block.rev_block.keep_input = True
            out = block(out, inverse=inverse)
        
        out = out + down
        return self.relu(out)


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, keep_input):
        super(UpTransition, self).__init__()
        self.up_conv_ab = self.build_up_conv(inChans, outChans)
        self.up_conv_ba = self.build_up_conv(inChans, outChans)

        self.core = nn.Sequential(*[VnetRevBlock(outChans, keep_input) for _ in range(nConvs)])
        self.relu = nn.PReLU(outChans)
    
    def build_up_conv(self, inChans, outChans):
        return nn.Sequential(nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2),
                             nn.BatchNorm3d(outChans // 2),
                             nn.PReLU(outChans // 2))

    def forward(self, x, skipx, inverse=False):
        if inverse:
            up_conv = self.up_conv_ba
            core = reversed(self.core)
        else:
            up_conv = self.up_conv_ab
            core = self.core

        up = up_conv(x)
        xcat = torch.cat((up, skipx), 1)
        out = xcat
        for i, block in enumerate(core):
            if i == 0:
                #https://github.com/silvandeleemput/memcnn/issues/39#issuecomment-599199122
                if inverse:
                    block.rev_block.keep_input_inverse = True
                else:
                    block.rev_block.keep_input = True
            out = block(out, inverse)

        out = out + xcat
        return self.relu(out)


class OutputTransition(nn.Module):
    def __init__(self, inChans, num_classes):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2) # should i put num_classes here as well?
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, num_classes, kernel_size=1)
        self.relu1 = nn.PReLU(num_classes)
        self.tanh = nn.Tanh() 

    def forward(self, x):
        # convolve 32 down to num of channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        res = self.tanh(out)
        return res


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, num_classes=1, keep_input=False):
        super(VNet, self).__init__()
        self.in_tr_ab = InputTransition(16)
        self.in_tr_ba = InputTransition(16)

        # ----- partially reversible layers ------

        # self.down_tr32 = DownTransition(16, 2, keep_input)
        # self.down_tr64 = DownTransition(32, 3, keep_input)
        # self.down_tr128 = DownTransition(64, 3, keep_input)
        # self.down_tr256 = DownTransition(128, 3, keep_input)

        # self.up_tr256 = UpTransition(256, 256, 3, keep_input)
        # self.up_tr128 = UpTransition(256, 128, 3, keep_input)
        # self.up_tr64 = UpTransition(128, 64, 2, keep_input)
        # self.up_tr32 = UpTransition(64, 32, 1, keep_input)

        self.down_tr32 = DownTransition(16, 1, keep_input)
        self.down_tr64 = DownTransition(32, 2, keep_input)
        self.down_tr128 = DownTransition(64, 3, keep_input)
        self.down_tr256 = DownTransition(128, 2, keep_input)

        self.up_tr256 = UpTransition(256, 256, 2, keep_input)
        self.up_tr128 = UpTransition(256, 128, 2, keep_input)
        self.up_tr64 = UpTransition(128, 64, 1, keep_input)
        self.up_tr32 = UpTransition(64, 32, 1, keep_input)
        
        # -----------------------------------------
        
        self.out_tr_ab = OutputTransition(32, num_classes)
        self.out_tr_ba = OutputTransition(32, num_classes)

    def forward(self, x, inverse=False):
        if inverse:
            in_tr  = self.in_tr_ba
            out_tr = self.out_tr_ba
        else:
            in_tr  = self.in_tr_ab
            out_tr = self.out_tr_ab
        
        out16 = in_tr(x)
        out32 = self.down_tr32(out16, inverse)
        out64 = self.down_tr64(out32, inverse)
        out128 = self.down_tr128(out64, inverse)
        out256 = self.down_tr256(out128, inverse)
        out = self.up_tr256(out256, out128, inverse)
        out = self.up_tr128(out, out64, inverse)
        out = self.up_tr64(out, out32, inverse)
        out = self.up_tr32(out, out16, inverse)
        out = out_tr(out)
        return out


class DeeperVNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, num_classes=1, keep_input=False):
        super(DeeperVNet, self).__init__()
        print('keep_input', keep_input)
        self.in_tr_ab = InputTransition(16)
        self.in_tr_ba = InputTransition(16)

        # ----- partially reversible layers ------

        self.down_tr32 = DownTransition(16, 2, keep_input)
        self.down_tr64 = DownTransition(32, 3, keep_input)
        self.down_tr128 = DownTransition(64, 3, keep_input)
        self.down_tr256 = DownTransition(128, 3, keep_input)

        self.up_tr256 = UpTransition(256, 256, 3, keep_input)
        self.up_tr128 = UpTransition(256, 128, 3, keep_input)
        self.up_tr64 = UpTransition(128, 64, 2, keep_input)
        self.up_tr32 = UpTransition(64, 32, 1, keep_input)
        
        # -----------------------------------------
        
        self.out_tr_ab = OutputTransition(32, num_classes)
        self.out_tr_ba = OutputTransition(32, num_classes)

    def forward(self, x, inverse=False):
        if inverse:
            in_tr  = self.in_tr_ba
            out_tr = self.out_tr_ba
        else:
            in_tr  = self.in_tr_ab
            out_tr = self.out_tr_ab
        
        out16 = in_tr(x)
        out32 = self.down_tr32(out16, inverse)
        out64 = self.down_tr64(out32, inverse)
        out128 = self.down_tr128(out64, inverse)
        out256 = self.down_tr256(out128, inverse)
        out = self.up_tr256(out256, out128, inverse)
        out = self.up_tr128(out, out64, inverse)
        out = self.up_tr64(out, out32, inverse)
        out = self.up_tr32(out, out16, inverse)
        out = out_tr(out)
        return out

