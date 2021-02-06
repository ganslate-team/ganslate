import logging
from pathlib import Path

from midaGAN.utils import communication
from midaGAN.utils.trackers import visuals_to_combined_2d_grid
from midaGAN.utils.trackers.base import BaseTracker


class TrainingTracker(BaseTracker):
    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logging.getLogger(type(self).__name__)
        self.log_freq = conf.train.logging.freq

    def log_iter(self, learning_rates, losses, visuals, metrics):
        """Parameters: # TODO: update this
            iters (int) -- current training iteration
            losses (tuple/list) -- training losses
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if self.iter_idx % self.log_freq == 0:
            visuals = {k: v for k, v in visuals.items() if v is not None}
            losses = {k: v for k, v in losses.items() if v is not None}
            metrics = {k: v for k, v in metrics.items() if v is not None}

            # reduce metrics (avg) and send to the process of rank 0
            metrics = communication.reduce(metrics, average=True, all_reduce=False)
            # reduce losses (avg) and send to the process of rank 0
            losses = communication.reduce(losses, average=True, all_reduce=False)

            self._log_message(learning_rates, losses)

            if communication.get_local_rank() == 0:
                visuals = visuals_to_combined_2d_grid(visuals)
                self._save_image(visuals, self.iter_idx)

                if self.wandb:
                    self.wandb.log_iter(self.iter_idx, learning_rates, losses, visuals, metrics)

                if self.tensorboard:
                    self.tensorboard.log_iter(self.iter_idx, learning_rates, losses, visuals,
                                              metrics)

    def _log_message(self, learning_rates, losses):
        message = '\n' + 20 * '-' + ' '

        if learning_rates:
            lr_G, lr_D = learning_rates["lr_G"], learning_rates["lr_D"]
            message += '(iter: %d | comp: %.3f, data: %.3f | lr_G: %.7f, lr_D = %.7f)' \
                        % (self.iter_idx, self.t_comp, self.t_data, lr_G, lr_D)
            message += ' ' + 20 * '-' + '\n'

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        self.logger.info(message)