import torch

from midaGAN.nn.gans.unpaired import cyclegan
from midaGAN import configs
from dataclasses import dataclass

@dataclass
class AdaptiveTrainingConfig:
    """Adaptive Training config that can be used for many different kinds of 
    adaptive procedures"""
    adaptive_lambda: bool = True
    adaptive_lr: bool = False
    
    change_rate: float = 0.05
    sample_size: int = 64*4
    
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
        self.max_lambda = 50

    def reset_sample_pool(self):
        self.sample_pool = []

    def init_optimizers(self):
        super().init_optimizers()
        self.init_adaptive_training()

    def backward_D(self, discriminator):
        super().backward_D(discriminator)

        if len(self.sample_pool) < self.sample_size:
            D_train = self.pred_real.clone()
            current_samples = [(_D_train > 0.5).float().mean() for _D_train in D_train]
            self.sample_pool.extend(current_samples)
        else:
            self.adapt_training()
        
    def adapt_training(self):
        eps = 0.05
        self.current_rt = torch.mean(torch.Tensor(self.sample_pool))
        self.metrics['rt'] = self.current_rt

        # If current rt is greater than set rt_factor, D
        # is potentially overfitting, decrease lambda rate so that
        # G can focus on adversarial loss.
        if self.current_rt > (self.rt_factor + eps):
            if self.adaptive_lambda:
                if self.criterion_G.lambda_A > self.max_lambda:
                    self.criterion_G.lambda_A = self.max_lambda 
                else:
                    self.criterion_G.lambda_A = self.criterion_G.lambda_A * (1 - self.change_rate)

                if self.criterion_G.lambda_B > self.max_lambda:
                    self.criterion_G.lambda_B = self.max_lambda 
                else:
                    self.criterion_G.lambda_B = self.criterion_G.lambda_B * (1 - self.change_rate)

        # If current rt is smaller than set rt_factor, D is not
        # discriminative, increase lambda rate so that G can 
        # focus on cycle consistency over adversarial till D catches up
        if self.current_rt < (self.rt_factor + eps):
            if self.adaptive_lambda:
                if self.criterion_G.lambda_A > self.max_lambda:
                    self.criterion_G.lambda_A = self.max_lambda 
                else:
                    self.criterion_G.lambda_A = self.criterion_G.lambda_A * (1 + self.change_rate)

                if self.criterion_G.lambda_B > self.max_lambda:
                    self.criterion_G.lambda_B = self.max_lambda 
                else:
                    self.criterion_G.lambda_B = self.criterion_G.lambda_B * (1 + self.change_rate)

     
        self.metrics['lambda_A'] = torch.tensor(self.criterion_G.lambda_A)
        self.metrics['lambda_B'] = torch.tensor(self.criterion_G.lambda_B)

        self.reset_sample_pool()

