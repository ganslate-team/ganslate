import logging
from pathlib import Path

import torch

from midaGAN.utils import communication
from midaGAN.utils.trackers.base import BaseTracker
from midaGAN.utils.trackers.utils import (process_visuals_for_logging,
                                          concat_batch_of_visuals_after_gather)


class ValTestTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logging.getLogger(type(self).__name__)

        self.metrics = []
        self.visuals = []

    def add_sample(self, visuals, metrics):
        """Parameters: # TODO: update this
        """
        metrics = {k: v for k, v in metrics.items() if v is not None}
        visuals = {k: v for k, v in visuals.items() if v is not None}

        # Reduce metrics (avg) and send to the process of rank 0
        metrics = communication.reduce(metrics, average=True, all_reduce=False)

        # Gather visuals from different processes to the rank 0 process
        visuals = communication.gather(visuals)
        visuals = concat_batch_of_visuals_after_gather(visuals)
        visuals = process_visuals_for_logging(visuals, single_example=False, grid_depth="mid")

        self.visuals.extend(visuals)
        self.metrics.append(metrics)

    def push_samples(self, iter_idx, prefix):
        """
        Push samples to start logging
        """
        for visuals_idx, visuals in enumerate(self.visuals):
            name = ""
            if prefix:
                name += f"{prefix}/"
            if iter_idx:
                name += f"{iter_idx}"
                # When val, put images in a dir for the iter at which it is validating.
                # When testing, there aren't multiple iters, so it isn't necessary.
                name += "/" if self.conf.mode == "val" else "_"
            name += f"{visuals_idx}"
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
                                visuals=self.visuals,
                                mode=mode,
                                metrics=averaged_metrics)

        # TODO: revisit tensorboard support
        if self.tensorboard:
            raise NotImplementedError("Tensorboard tracking not implemented")
            # self.tensorboard.log_iter(iter_idx, None, None, visuals, metrics)

        # Clear stored buffer after pushing the results
        self.metrics = []
        self.visuals = []

    def _log_message(self, index, metrics, prefix=""):
        message = '\n' + 20 * '-' + ' '

        message += f"({self.conf.mode.capitalize()}"

        if index is not None:
            message += f" at iter {index}"
        if prefix != "":
            message += f" for dataset '{prefix}'"

        message += ') ' + 20 * '-' + '\n'

        for k, v in metrics.items():
            name = f'{prefix}_{k}' if prefix != "" else str(k)
            message += '%s: %.3f ' % (name, v)

        self.logger.info(message)
