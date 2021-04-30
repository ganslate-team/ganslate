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
    change_mode: str = "threshold"

    patience: int = 100
    freq: int = 100
    warmup_iter: int = 20


@dataclass
class AdaptiveCycleGANConfig(cyclegan.CycleGANConfig):
    """CycleGAN Config"""
    name: str = "AdaptiveCycleGAN"
    adaptive_training: AdaptiveTrainingConfig = AdaptiveTrainingConfig()


class AdaptiveCycleGAN(cyclegan.CycleGAN):
    def init_adaptive_training(self):
        self.iter = 1

        self.sample_pool = []
        self.adaptive_training_config = self.conf[self.conf.mode].gan.adaptive_training
        
        # Check what adaptive mode is enabled.
        self.adaptive_lambda = self.adaptive_training_config.adaptive_lambda
        self.adaptive_lr = self.adaptive_training_config.adaptive_lr
        
        self.change_rate = self.adaptive_training_config.change_rate
        self.change_mode = self.adaptive_training_config.change_mode

        self.patience = self.adaptive_training_config.patience
        self.freq = self.adaptive_training_config.freq
        self.warmup_iter = self.adaptive_training_config.warmup_iter

        self.max_lambda = 50

    def reset_sample_pool(self):
        self.sample_pool = []

    def init_optimizers(self):
        super().init_optimizers()
        self.init_adaptive_training()

    def optimize_parameters(self):
        super().optimize_parameters()

        if self.iter > self.warmup_iter:
            if self.iter % self.freq == 0:
                self.adapt_training()
                self.reset_sample_pool()

            else:
                # Keep only as many samples as in the patience parameter
                if self.iter % self.patience:
                    self.reset_sample_pool()

                current_sample = {
                    "cycle_A": self.losses['cycle_A'],
                    "cycle_B": self.losses['cycle_B']
                }

                self.sample_pool.append(current_sample)

        self.iter += 1


    def adapt_training(self):
        print("Samples=", len(self.sample_pool))
        min_cycle_A = np.min([loss["cycle_A"] for loss in self.sample_pool])
        min_cycle_B = np.min([loss["cycle_B"] for loss in self.sample_pool])

        # Check the aggregated sample pool for
        if self.losses['cycle_A'] >= min_cycle_A:
            if self.change_mode == "threshold":
                self.criterion_G.lambda_AB = self.criterion_G.lambda_AB * (1 + self.change_rate)

            elif self.change_mode == "relative":
                change_rate = (self.prev_sample['cycle_A'] - self.losses['cycle_A'])/self.prev_sample['cycle_A']
                self.criterion_G.lambda_AB = self.criterion_G.lambda_AB * (1 + change_rate)

        if self.losses['cycle_B'] >= min_cycle_B:
            if self.change_mode == "threshold":
                self.criterion_G.lambda_BA = self.criterion_G.lambda_BA * (1 + self.change_rate)

            elif self.change_mode == "relative":
                change_rate = (self.prev_sample['cycle_B'] - self.losses['cycle_B'])/self.prev_sample['cycle_B']
                self.criterion_G.lambda_BA = self.criterion_G.lambda_BA * (1 + change_rate)

        # Clip lambda values between 0 and max_lambda parameter
        self.criterion_G.lambda_AB = np.clip(self.criterion_G.lambda_AB, 0, self.max_lambda)
        self.criterion_G.lambda_BA = np.clip(self.criterion_G.lambda_BA, 0, self.max_lambda)

        self.metrics['lambda_AB'] = torch.tensor(self.criterion_G.lambda_AB)
        self.metrics['lambda_BA'] = torch.tensor(self.criterion_G.lambda_BA)
        print(f"Adapting training with lambda_AB={self.metrics['lambda_AB']} and lambda_BA={self.metrics['lambda_BA']} ")


