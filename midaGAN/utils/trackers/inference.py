#from pathlib import Path
import logging
import time

from midaGAN.utils import communication
from midaGAN.utils.trackers.base import BaseTracker


class InferenceTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logging.getLogger(type(self).__name__)

    def log_message(self, iter_idx, data_loader):
        iter_idx *= communication.get_world_size()
        # Get the len of dataset, not dataloader, since each DDP loader only has a portion of the dataset
        len_dataset = len(data_loader.dataset)
        # In case of DDP, if (len_dataset % number of processes != 0), it will show more iters than there are
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
        # reduce computational time data point (avg) and send to the process of rank 0
        self.t_save = communication.reduce(self.t_save, average=True, all_reduce=False)
