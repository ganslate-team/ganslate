import torch
import midaGAN.nn.losses.ssim as ssim
from midaGAN.nn.utils import reshape_to_4D_if_5D


class EvaluationMetrics:
    def __init__(self, conf):
        self.conf = conf
        
    def SSIM(self, input, target):
        channels_ssim = input.shape[2]

        ssim_loss = ssim.SSIM(data_range=1, 
                                    channel=channels_ssim, nonnegative_ssim=True)
        # Gradient computation not needed for metric computation
        input = input.detach()
        target = target.detach()

        input = reshape_to_4D_if_5D(input)
        target = reshape_to_4D_if_5D(target)

        # Data range needs to be positive and normalized
        # https://github.com/VainF/pytorch-msssim#2-normalized-input
        input =  (input + 1)/2
        target = (target + 1)/2

        return 1 - ssim_loss(input, target)
    
        
    # def get_metrics(self, input, target):
    #     metrics = {}

    #     if self.conf.metrics.ssim:
    #         metrics.update({
    #             'SSIM': SSIM(input, target)
    #         })

    #     return metrics
    