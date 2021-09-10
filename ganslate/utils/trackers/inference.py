#from pathlib import Path
from loguru import logger
import time

import torch

from ganslate.utils import communication
from ganslate.utils.trackers.base import BaseTracker
from ganslate.utils.trackers.utils import (process_visuals_for_logging,
                                          concat_batch_of_visuals_after_gather)


class InferenceTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logger

    def log_iter(self, visuals, len_dataset):
        
        def parse_visuals(visuals):
            # Gather visuals from different processes to the rank 0 process
            visuals = communication.gather(visuals)
            visuals = concat_batch_of_visuals_after_gather(visuals)
            visuals = process_visuals_for_logging(self.conf, visuals, single_example=False)
            return visuals
        
        def log_message():
            # In case of DDP, if (len_dataset % number of processes != 0),
            # it will show more iters than there actually are
            if self.iter_idx > len_dataset:
                self.iter_idx = len_dataset

            message = (f"{self.iter_idx}/{len_dataset} - loading: {self.t_data:.2f}s",
                       f" | inference: {self.t_comp:.2f}s | saving: {self.t_save:.2f}s")
            self.logger.info(message)

        visuals = parse_visuals(visuals)
        log_message()

        for i, visuals_grid in enumerate(visuals):
            # In DDP, each process is for a different iter, so incrementing it accordingly
            self._save_image(visuals_grid, self.iter_idx + i)

            if self.wandb:
                self.wandb.log_iter(iter_idx=self.iter_idx + i,
                                    visuals=visuals_grid,
                                    mode="infer")

            if self.tensorboard:
                self.tensorboard.log_iter(iter_idx=self.iter_idx + i,
                                          visuals=visuals_grid,
                                          mode="infer")


    def start_saving_timer(self):
        self.saving_start_time = time.time()

    def end_saving_timer(self):
        self.t_save = (time.time() - self.saving_start_time) / self.batch_size
        # Reduce computational time data point (avg) and send to the process of rank 0
        self.t_save = communication.reduce(self.t_save, average=True, all_reduce=False)
