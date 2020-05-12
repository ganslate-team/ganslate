
import os
import logging
import time

import torch
from torchvision.utils import save_image

from util.distributed import comm, multi_gpu
from util.file_utils import mkdirs

import wandb
from torch.utils.tensorboard import SummaryWriter


class ExperimentTracker:
    def __init__(self, conf):
        self.output_dir = conf.logging.output_dir
        mkdirs(os.path.join(self.output_dir, 'images'))
        self._save_config(conf)

        self.batch_size = conf.batch_size
        self.total_samples = 0
        self.iter_idx = None
        self.iter_end_time = None
        self.iter_start_time = None
        self.t_data = None
        self.t_comp = None

        # TODO: initialize these if defined by conf
        self.wandb = WandbTracker(conf)
        self.tensorboard = TensorboardTracker(conf)

    def _save_config(self, conf):
        config_path = os.path.join(self.output_dir, "config.yaml")
        with open(config_path, "w") as file:
            file.write(conf.pretty())

    def set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx

    def start_computation_timer(self):
        self.iter_start_time = time.time()

    def start_dataloading_timer(self):
        self.iter_end_time = time.time()
    
    def end_computation_timer(self):
        self.t_comp = (time.time() - self.iter_start_time) / self.batch_size
        # reduce computational time data point (avg) and send to the process of rank 0
        self.t_comp = comm.reduce(self.t_comp, average=True, all_reduce=False)

    def end_dataloading_timer(self):
        self.t_data = self.iter_start_time - self.iter_end_time # is it per sample or per batch?
        # reduce data loading per data point (avg) and send to the process of rank 0
        self.t_data = comm.reduce(self.t_data, average=True, all_reduce=False)

    def log_iter(self, learning_rates, losses, visuals): # TODO: implement visuals, tensorboard, wandb, python logger
        """Parameters: # TODO: update this
            iters (int) -- current training iteration
            losses (tuple/list) -- training losses
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """        
        visuals = {k: v for k, v in visuals.items() if v != None}
        losses = {k: v for k, v in losses.items() if v != None}
        losses = comm.reduce(losses, average=True, all_reduce=False) # reduce losses (avg) and send to the process of rank 0

        if multi_gpu.is_main_process():
            self.total_samples += self.batch_size * multi_gpu.get_world_size()
            self._log_message(learning_rates, losses) 

            visuals = self._visuals_to_combined_2d_grid(visuals)
            self._save_image(visuals)

            self.wandb.log_iter(self.iter_idx, learning_rates, losses, visuals)
            self.tensorboard.log_iter(self.iter_idx, learning_rates, losses, visuals)

    def _log_message(self, learning_rates, losses):
        lr_G, lr_D = learning_rates
        message = '\n' + 20 * '-' + ' '
        message += '(iter: %d, samples: %d | comp: %.3f, data: %.3f | lr_G: %.7f, lr_D = %.7f)' \
                       % (self.iter_idx, self.total_samples, self.t_comp, self.t_data, lr_G, lr_D)
        message += ' ' + 20 * '-' +  '\n'
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
    
    def _visuals_to_combined_2d_grid(self, visuals):
        # Concatenate slices that are at the same level from different visuals along width (each tensor from visuals.values() is NxCxDxHxW, hence dim=4)
        combined_slices = torch.cat(visuals.values(), dim=4) 
        combined_slices = combined_slices[0] # we plot a single volume from the batch
        combined_slices = combined_slices.permute(1,0,2,3) # CxDxHxW -> DxCxHxW

        # Concatenate all combined slices along height to form a single 2d image (the tensor in tuple are CxHxW, hence dim=1)
        combined_image = torch.cat(tuple(combined_slices), dim=1) 
        combined_image = (combined_image + 1) / 2 # [-1,1] -> [0,1]. Data range important when saving images.

        name = "-".join(visuals.keys()) # e.g. "real_A-fake_B-rec_A-real_B-fake_A-rec_B"
        return {'name': name, 'image': combined_image} # NOTE: image format is CxHxW

    def _save_image(self, visuals):
        name, image = visuals['name'], visuals['image']
        file_path = os.path.join(self.output_dir, 'images/%d_%s.png' % (self.iter_idx, name))
        save_image(image, file_path)


class WandbTracker:
    def __init__(self, conf):
        wandb.init(project="my-project", config=dict(conf)) # TODO: project and organization from conf

    def log_iter(self, iter_idx, learning_rates, losses, visuals):
        """TODO"""
        log_dict = {}

        # Learning rates
        lr_G, lr_D = learning_rates
        log_dict['lr_G'] = lr_G
        log_dict['lr_D'] = lr_D
        
        # Losses
        for name, loss in losses.items():
            log_dict['loss_%s' % name] = loss
        
        # Image
        name, image = visuals['name'], visuals['image']
        image = image.permute(1,2,0) # CxHxW -> HxWxC
        log_dict[name] = [wandb.Image(image.cpu().detach().numpy())]

        wandb.log(log_dict)


class TensorboardTracker:
    def __init__(self, conf):
        self.writer = SummaryWriter(conf.logging.output_dir)

    def __del__(self):
        self.writer.close()

    def log_iter(self, iter_idx, learning_rates, losses, visuals):
        # Learning rates
        lr_G, lr_D = learning_rates
        self.writer.add_scalar('Learning Rates/lr_G', lr_G, iter_idx)
        self.writer.add_scalar('Learning Rates/lr_D', lr_D, iter_idx)

        # Losses
        for name, loss in losses.items():
            self.writer.add_scalar('Losses/%s' % name, loss, iter_idx)

        # Image
        name, image = visuals['name'], visuals['image']
        self.writer.add_image('Visuals/%s' % name, image, iter_idx, dataformats='CHW')