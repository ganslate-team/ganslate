import logging
import time
from pathlib import Path

import torchvision
from midaGAN.utils import communication, io
from midaGAN.utils.trackers import visuals_to_combined_2d_grid
from midaGAN.utils.trackers.base_tracker import BaseTracker
from midaGAN.utils.trackers.tensorboard_tracker import TensorboardTracker
from midaGAN.utils.trackers.wandb_tracker import WandbTracker


class EvalTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logging.getLogger(type(self).__name__)
        self.wandb = None
        self.tensorboard = None
        if communication.get_local_rank() == 0:
            io.mkdirs(Path(self.output_dir) / 'eval_images')
            if conf.logging.wandb:
                self.wandb = WandbTracker(conf)
            if conf.logging.tensorboard:
                self.tensorboard = TensorboardTracker(conf)

    def start_saving_timer(self):
        self.saving_start_time = time.time()

    def end_saving_timer(self):
        self.t_save = (time.time() - self.saving_start_time) / self.batch_size
        # reduce computational time data point (avg) and send to the process of rank 0
        self.t_save = communication.reduce(self.t_save, average=True, all_reduce=False)

    def _log_message(self, index, metrics):
        message = '\n' + 20 * '-' + ' '
        message += f'(sample: {index})'
        message += ' ' + 20 * '-' + '\n'

        for k, v in metrics.items():
            message += '%s: %.3f ' % (k, v)

        self.logger.info(message)

    def log_sample(self, train_index, index, visuals, metrics):
        """Parameters: # TODO: update this
        """
        self.iter_idx = index
        visuals = {k: v for k, v in visuals.items() if v is not None}
        metrics = {k: v for k, v in metrics.items() if v is not None}

        metrics = communication.reduce(
            metrics, average=True,
            all_reduce=False)  # reduce metrics (avg) and send to the process of rank 0

        self._log_message(self.iter_idx, metrics)

        if communication.get_local_rank() == 0:
            visuals = visuals_to_combined_2d_grid(visuals)
            self._save_image(visuals)

            if self.wandb:
                self.wandb.log_iter(train_index, {}, {}, visuals, metrics, batch=index)

            if self.tensorboard:
                self.tensorboard.log_iter(train_index, {}, {}, visuals, metrics, batch=index)

    def _save_image(self, visuals):
        name, image = visuals['name'], visuals['image']
        file_path = Path(self.output_dir) / f"eval_images/{self.iter_idx}_{name}.png"
        torchvision.utils.save_image(image, file_path)
