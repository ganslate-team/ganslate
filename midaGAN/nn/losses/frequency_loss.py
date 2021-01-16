import torch
import torch.nn as nn
import cv2

import numpy as np

import logging
logger = logging.getLogger(__name__)


class FrequencyLoss:
    def __init__(self, conf):
        lambda_A = conf.gan.optimizer.lambda_A
        lambda_B = conf.gan.optimizer.lambda_B
        lambda_F = conf.gan.optimizer.lambda_F
        patch_size = conf.dataset.patch_size
        spectrum = conf.gan.optimizer.spectrum
        filt = conf.gan.optimizer.filt
        freq_distance_metric = conf.gan.optimizer.freq_distance_metric
        adversarial = conf.gan.optimizer.freq_adversarial

        if lambda_F > 0:
            self.criterion_freq = FreqLoss(lambda_A, lambda_B, lambda_F, patch_size, spectrum, filt, freq_distance_metric, adversarial)
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
            losses['freq_A'], losses['freq_B'] = self.criterion_freq(real_A, real_B, fake_A, fake_B, rec_A, rec_B)

        return losses

class FreqLoss:
    def __init__(self, lambda_A, lambda_B, lambda_F, patch_size, spectrum, filt, distance_metric, adversarial):
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_F = lambda_F
        self.filter = filt
        self.spectrum = spectrum
        self.criterion = torch.nn.L1Loss()
        self.patch_size = patch_size
        self.distance_metric = distance_metric
        self.adversarial = adversarial
        self.hamming_filter()
        self.lowpass_filter()
        self.highpass_filter()

    def apply_distance_metric(self, ft_arr):
        if self.distance_metric == "log":
            ft_arr = np.log(ft_arr)
        elif self.distance_metric == "euclidean":
            # division by patch size and multiplication by 10000 to reach same order of magnitude as log
            ft_arr = ft_arr/(self.patch_size[0]*self.patch_size[1]*self.patch_size[2])*10000
        elif self.distance_metric == "euclidean_squared":
            # division by patch size to reach same order of magnitude as log
            ft_arr = ft_arr**2/(self.patch_size[0]*self.patch_size[1]*self.patch_size[2])
        return ft_arr

    def lowpass_filter(self):
        lowp_filter = np.zeros((self.patch_size[1], self.patch_size[2]))
        center_coordinates = (int(self.patch_size[1]/2), int(self.patch_size[2]/2))
        # threshold defined by frequency distance experiments between real MR and corresponding co-registered real CT
        threshold = 41
        self.lowp_filter = cv2.circle(lowp_filter, center_coordinates, threshold, color=1, thickness=-1, lineType=8, shift=0)
        return
    
    def highpass_filter(self):
        highp_filter = np.ones((self.patch_size[1], self.patch_size[2]))
        center_coordinates = (int(self.patch_size[1]/2), int(self.patch_size[2]/2))
        # threshold defined by frequency distance experiments between real MR and corresponding co-registered real CT
        threshold = 20
        self.highp_filter = cv2.circle(highp_filter, center_coordinates, threshold, color=0, thickness=-1, lineType=8, shift=0)
        return
    
    def hamming_filter(self):
        h_x = np.hamming(self.patch_size[1])
        h_y = np.hamming(self.patch_size[2])
        ham2d = np.sqrt(np.outer(h_x, h_y))
        # Note: patch_size[0] corresponds to z_size
        ham3d = np.tile(ham2d, (self.patch_size[0], 1, 1))
        self.ham3d = np.reshape(ham3d, (1,1,self.patch_size[0],self.patch_size[1],self.patch_size[2]))
        return

    def apply_filter(self, ft_arr):
        if self.filter == "hamming":
            filtered_arr = np.multiply(ft_arr, self.ham3d)
        elif self.filter == "lowpass":
            filtered_arr = np.multiply(ft_arr, self.lowp_filter)
        elif self.filter == "highpass":
            filtered_arr = np.multiply(ft_arr, self.highp_filter)
        return filtered_arr
    
    def get_fft_spectrum(self, img):
        # First copy tensor to host memory via .cpu()
        # Then detach and convert to numpy to be able to apply ft
        fft_spectrum = np.fft.fftn(img.cpu().detach().numpy(), axes=(2, 3, 4))
        # Shift so that ft is centered around origin
        fft_spectrum = np.fft.fftshift(fft_spectrum)
        if self.spectrum == "magnitude":
            fft_spectrum = np.abs(fft_spectrum)
        elif self.spectrum == "phase":
            # shift by pi to get angle values between 0 and 2 pi
            # multiplicative factor of 3 to move values into same range as magnitude spectrum
            fft_spectrum = (np.angle(fft_spectrum) + np.pi)
        fft_spectrum = self.apply_distance_metric(fft_spectrum)
        fft_spectrum = self.apply_filter(fft_spectrum)
        return fft_spectrum
    
    def __call__(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        spectrum_real_A = self.get_fft_spectrum(real_A)
        spectrum_real_B = self.get_fft_spectrum(real_B)
        if self.adversarial:
            spectrum_rec_A = self.get_fft_spectrum(rec_A)
            spectrum_rec_B = self.get_fft_spectrum(rec_B)
        else:
            spectrum_fake_A = self.get_fft_spectrum(fake_A)
            spectrum_fake_B = self.get_fft_spectrum(fake_B)
        

        # to properly counterbalance the filter weights
        if self.filter == "hamming":
            mult_factor = 1/(np.sum(self.ham3d)/(spectrum_real_A.size))
        else:
            nonzero_els = np.count_nonzero(spectrum_real_A)
            mult_factor = spectrum_real_A.size/nonzero_els

        # need to convert back to torch tensor to apply loss
        if self.adversarial:
            loss_freq_A = self.criterion(torch.from_numpy(spectrum_rec_A), torch.from_numpy(spectrum_real_A))
            loss_freq_B = self.criterion(torch.from_numpy(spectrum_rec_B), torch.from_numpy(spectrum_real_B))
        else:
            loss_freq_A = self.criterion(torch.from_numpy(spectrum_fake_B), torch.from_numpy(spectrum_real_A))
            loss_freq_B = self.criterion(torch.from_numpy(spectrum_fake_A), torch.from_numpy(spectrum_real_B))

        loss_freq_A = loss_freq_A * self.lambda_B * self.lambda_F * mult_factor
        loss_freq_B = loss_freq_B * self.lambda_A * self.lambda_F * mult_factor
        return loss_freq_A, loss_freq_B