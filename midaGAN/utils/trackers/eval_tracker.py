import logging
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

        self.metrics = []
        self.visuals = []

    def _log_message(self, index, metrics):
        message = '\n' + 20 * '-' + ' '
        message += f'(sample: {index})'
        message += ' ' + 20 * '-' + '\n'

        for k, v in metrics.items():
            message += '%s: %.3f ' % (k, v)

        self.logger.info(message)

    def add_sample(self, visuals, metrics):
        """Parameters: # TODO: update this
        """
        visuals = {k: v for k, v in visuals.items() if v is not None}
        metrics = {k: v for k, v in metrics.items() if v is not None}

        metrics = communication.reduce(
            metrics, average=True,
            all_reduce=False)  # reduce metrics (avg) and send to the process of rank 0

        if communication.get_local_rank() == 0:
            visuals = visuals_to_combined_2d_grid(visuals)
            self.visuals.append(visuals)
            self.metrics.append(metrics)

    def push_samples(self, iter_idx):
        """
        Push samples to start logging
        """
        if communication.get_local_rank() == 0:
            print("Number of samples", len(self.visuals))
            for visuals in self.visuals:
                self._save_image(visuals)

            # Averages list of dictionaries within self.metrics
            averaged_metrics = {
                key: sum(metric[key] for metric in self.metrics) / len(self.metrics)
                for key in self.metrics[0]
            }

            self._log_message(iter_idx, averaged_metrics)

            if self.wandb:
                self.wandb.log_iter(iter_idx, {}, {},
                                    self.visuals,
                                    averaged_metrics,
                                    mode='validation')
            #TODO: Adapt eval tracker for tensorboard
            if self.tensorboard:
                raise NotImplementedError("Tensorboard evaluation tracking not implemented")
                # self.tensorboard.log_iter(iter_idx, {}, {}, visuals, metrics)

            # Clear stored buffer after pushing the results
            self.metrics = []
            self.visuals = []

    def _save_image(self, visuals):
        name, image = visuals['name'], visuals['image']
        file_path = Path(self.output_dir) / f"eval_images/{self.iter_idx}_{name}.png"
        torchvision.utils.save_image(image, file_path)
