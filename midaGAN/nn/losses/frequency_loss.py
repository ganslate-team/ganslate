import torch
import torch.nn as nn

import numpy as np

import logging
logger = logging.getLogger(__name__)

# TODO: 2D ft loss
# TODO: adversarial ft loss
# TODO: additional filters
# TODO: thresholding/manipulating (via log for example) ft

class FrequencyLoss:
    def __init__(self, conf):
        lambda_A = conf.gan.optimizer.lambda_A
        lambda_B = conf.gan.optimizer.lambda_B
        lambda_F = conf.gan.optimizer.lambda_F
        patch_size = conf.dataset.patch_size

        if lambda_F > 0:
            self.criterion_freq = FreqLoss(lambda_A, lambda_B, lambda_F, patch_size)
        else:
            self.criterion_freq =  None

    def __call__(self, visuals):
        real_A, real_B = visuals['real_A'], visuals['real_B']
        fake_A, fake_B = visuals['fake_A'], visuals['fake_B']
        rec_A,  rec_B  = visuals['rec_A'],  visuals['rec_B']
        idt_A,  idt_B  = visuals['idt_A'],  visuals['idt_B']
        
        losses = {}

        # frequency loss
        if self.criterion_freq:
            losses['freq_A'], losses['freq_B'] = self.criterion_freq(real_A, real_B, fake_A, fake_B)

        return losses


class FreqLoss:
    def __init__(self, lambda_A, lambda_B, lambda_F, patch_size):
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_F = lambda_F
        self.criterion = torch.nn.L1Loss()
        self.patch_size = patch_size
        self.hamming_filter()

    def filter_a():
        return
    
    def hamming_filter(self):
        h_x = np.hamming(self.patch_size[1])
        h_y = np.hamming(self.patch_size[2])
        ham2d = np.sqrt(np.outer(h_x, h_y))
        ham3d = np.tile(ham2d, (self.patch_size[0], 1, 1))
        self.ham3d = np.reshape(ham3d, (1,1,self.patch_size[0],self.patch_size[1],self.patch_size[2]))
        return

    def apply_filter(self, ft_arr):
        filtered_arr = np.multiply(ft_arr, self.ham3d)
        return filtered_arr
    
    def __call__(self, real_A, real_B, fake_A, fake_B):
        # First copy tensor to host memory via .cpu()
        # Then detach and convert to numpy to be able to apply ft
        ft_real_A = np.fft.fftn(real_A.cpu().detach().numpy(), axes=(2, 3, 4))
        # Shift so that ft is centered around origin
        ft_real_A = np.fft.fftshift(ft_real_A)
        # log is used because the dynamic range of the Fourier coefficients is very large
        logmags_real_A = np.log(np.abs(ft_real_A))
        logmags_real_A = self.apply_filter(logmags_real_A)

        ft_fake_B = np.fft.fftn(fake_B.cpu().detach().numpy(), axes=(2, 3, 4))
        ft_fake_B = np.fft.fftshift(ft_fake_B)
        logmags_fake_B = np.log(np.abs(ft_fake_B))
        logmags_fake_B = self.apply_filter(logmags_fake_B)

        ft_real_B = np.fft.fftn(real_B.cpu().detach().numpy(), axes=(2, 3, 4))
        ft_real_B = np.fft.fftshift(ft_real_B)
        logmags_real_B = np.log(np.abs(ft_real_B))
        logmags_real_B = self.apply_filter(logmags_real_B)

        ft_fake_A = np.fft.fftn(fake_A.cpu().detach().numpy(), axes=(2, 3, 4))
        ft_fake_A = np.fft.fftshift(ft_fake_A)
        logmags_fake_A = np.log(np.abs(ft_fake_A))
        logmags_fake_A = self.apply_filter(logmags_fake_A)

        # need to convert back to torch tensor to apply loss
        loss_freq_A = self.criterion(torch.from_numpy(logmags_fake_B), torch.from_numpy(logmags_real_A))
        loss_freq_B = self.criterion(torch.from_numpy(logmags_fake_A), torch.from_numpy(logmags_real_B))

        loss_freq_A = loss_freq_A * self.lambda_B * self.lambda_F
        loss_freq_B = loss_freq_B * self.lambda_A * self.lambda_F
        return loss_freq_A, loss_freq_B