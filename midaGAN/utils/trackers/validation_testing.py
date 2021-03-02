import logging
from pathlib import Path

from midaGAN.utils import communication
from midaGAN.utils.trackers.utils import visuals_to_combined_2d_grid
from midaGAN.utils.trackers.base import BaseTracker


class ValTestTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logging.getLogger(type(self).__name__)

        self.metrics = []
        self.visuals = []

    def add_sample(self, visuals, metrics):
        """Parameters: # TODO: update this
        """
        visuals = {k: v for k, v in visuals.items() if v is not None}
        metrics = {k: v for k, v in metrics.items() if v is not None}

        # reduce metrics (avg) and send to the process of rank 0
        metrics = communication.reduce(metrics, average=True, all_reduce=False)

        if communication.get_local_rank() == 0:
            visuals = visuals_to_combined_2d_grid(visuals, grid_depth='mid')
            self.visuals.append(visuals)
            self.metrics.append(metrics)

    def push_samples(self, iter_idx, prefix=''):
        """
        Push samples to start logging
        """
        if communication.get_local_rank() == 0:

            for idx, visuals in enumerate(self.visuals):
                name = f"{prefix}_{iter_idx}_{idx}" if prefix != "" else f"{iter_idx}_{idx}"
                self._save_image(visuals, name)

            # Averages list of dictionaries within self.metrics
            averaged_metrics = {}
            # Each element of self.metrics has the same metric keys so
            # so fetching the key names from self.metrics[0] is fine
            for key in self.metrics[0]:
                metric_average = sum(metric[key] for metric in self.metrics) / len(self.metrics)
                averaged_metrics[key] = metric_average

            self._log_message(iter_idx, averaged_metrics, prefix=prefix)

            if self.wandb:
                mode = prefix if prefix != "" else self.conf.mode
                self.wandb.log_iter(iter_idx=iter_idx,
                                    learning_rates={},
                                    losses={},
                                    visuals=self.visuals,
                                    metrics=averaged_metrics,
                                    mode=prefix)

            # TODO: Adapt eval tracker for tensorboard
            if self.tensorboard:
                raise NotImplementedError("Tensorboard eval tracking not implemented")
                # self.tensorboard.log_iter(iter_idx, {}, {}, visuals, metrics)

            # Clear stored buffer after pushing the results
            self.metrics = []
            self.visuals = []

    def _log_message(self, index, metrics, prefix=''):
        message = '\n' + 20 * '-' + ' '
        message += f'({self.conf.mode} at iter {index})'
        message += ' ' + 20 * '-' + '\n'

        for k, v in metrics.items():
            name = f'{prefix}_{k}' if prefix != "" else str(k)
            message += '%s: %.3f ' % (name, v)

        self.logger.info(message)
