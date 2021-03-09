#from pathlib import Path
import logging
import time

from midaGAN.utils import communication
from midaGAN.utils.trackers.base import BaseTracker
from midaGAN.utils.trackers.utils import visuals_to_combined_2d_grid


class InferenceTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logging.getLogger(type(self).__name__)

    def log_iter(self, visuals, len_dataset):
        self._log_message(len_dataset)
        visuals = visuals_to_combined_2d_grid(visuals)
        self._save_image(visuals, self.iter_idx)

        if self.wandb:
            self.wandb.log_iter(iter_idx=self.iter_idx,
                                visuals=visuals,
                                mode=self.conf.mode)

        # TODO: revisit tensorboard support
        if self.tensorboard:
            raise NotImplementedError("Tensorboard tracking not implemented")
            # self.tensorboard.log_iter(self.iter_idx, learning_rates, losses, visuals,
            #                             metrics, self.conf.mode)
    
    def _log_message(self, len_dataset):
        iter_idx = self.iter_idx * communication.get_world_size()

        # In case of DDP, if (len_dataset % number of processes != 0),
        # it will show more iters than there actually are
        if iter_idx > len_dataset:
            iter_idx = len_dataset

        message = f"{iter_idx}/{len_dataset} - loading: {self.t_data:.2f}s"
        message += f" | inference: {self.t_comp:.2f}s"
        message += f" | saving: {self.t_save:.2f}s"
        self.logger.info(message)

    def start_saving_timer(self):
        self.saving_start_time = time.time()

    def end_saving_timer(self):
        self.t_save = (time.time() - self.saving_start_time) / self.batch_size
        # Reduce computational time data point (avg) and send to the process of rank 0
        self.t_save = communication.reduce(self.t_save, average=True, all_reduce=False)
