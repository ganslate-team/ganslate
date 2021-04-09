import torch

from midaGAN.nn.gans.unpaired import cyclegan
from midaGAN import configs
from dataclasses import dataclass
import numpy as np

@dataclass
class AdaptiveTrainingConfig:
    """Adaptive Training config that can be used for many different kinds of 
    adaptive procedures"""
    adaptive_lambda: bool = True
    adaptive_lr: bool = False
    
    change_rate: float = 0.01

    # Number of samples to consider for the rt factor
    sample_size: int = 256
    freq: int = 1000

    # TODO: Add link to paper where rt is described
    rt: float = 0.6


@dataclass
class AdaptiveCycleGANConfig(cyclegan.CycleGANConfig):
    """CycleGAN Config"""
    name: str = "AdaptiveCycleGAN"
    adaptive_training: AdaptiveTrainingConfig = AdaptiveTrainingConfig()


class AdaptiveCycleGAN(cyclegan.CycleGAN):
    def init_adaptive_training(self):
        self.sample_pool = []
        self.adaptive_training_config = self.conf[self.conf.mode].gan.adaptive_training
        self.sample_size = self.adaptive_training_config.sample_size
        
        # Check what adaptive mode is enabled.
        self.adaptive_lambda = self.adaptive_training_config.adaptive_lambda
        self.adaptive_lr = self.adaptive_training_config.adaptive_lr
        
        self.change_rate = self.adaptive_training_config.change_rate
        self.rt_factor = self.adaptive_training_config.rt
        self.freq = self.adaptive_training_config.freq
        self.max_lambda = 50

    def reset_sample_pool(self):
        self.sample_pool = []

    def init_optimizers(self):
        super().init_optimizers()
        self.init_adaptive_training()
        self.iter = 1

    def backward_D(self, discriminator):
        super().backward_D(discriminator)

        self.iter += 1
        D_generated = self.pred_fake.clone()
        n_samples = len(D_generated)
        if len(self.sample_pool) < self.sample_size:
            current_samples = [(_D_generated > 0.5).float().mean() for _D_generated in D_generated]
            self.sample_pool.extend(current_samples)
        elif self.iter != self.freq:
            del self.sample_pool[0:n_samples]
        else:
            print(f"Adapting training with {len(self.sample_pool)}")
            self.adapt_training()
            self.iter = 1
        
    def adapt_training(self):
        self.current_rt = torch.mean(torch.Tensor(self.sample_pool))
        self.metrics['rt'] = self.current_rt

        # If current rt is greater than set rt_factor, D
        # is largely predicting generated voxels as real,
        # increase lambda rate so that G focuses less on adversarial
        if self.current_rt > self.rt_factor:
            if self.adaptive_lambda:
                self.criterion_G.lambda_A = self.criterion_G.lambda_A * (1 + self.change_rate)
                self.criterion_G.lambda_B = self.criterion_G.lambda_B * (1 + self.change_rate)

        # If current rt is smaller than set rt_factor, D is not
        # discriminative, decrease lambda rate so that G can 
        # focus on adversarial till D catches up
        if self.current_rt < self.rt_factor:
            if self.adaptive_lambda:
                self.criterion_G.lambda_A = self.criterion_G.lambda_A * (1 - self.change_rate)
                self.criterion_G.lambda_B = self.criterion_G.lambda_B * (1 - self.change_rate)

        # Clip lambda values between 0 and max_lambda parameter
        self.criterion_G.lambda_A = np.clip(self.criterion_G.lambda_A, 0, self.max_lambda)
        self.criterion_G.lambda_B = np.clip(self.criterion_G.lambda_B, 0, self.max_lambda)

        self.metrics['lambda_A'] = torch.tensor(self.criterion_G.lambda_A)
        self.metrics['lambda_B'] = torch.tensor(self.criterion_G.lambda_B)

        self.reset_sample_pool()

