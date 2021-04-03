from loguru import logger
from pathlib import Path

import torch

from midaGAN.utils import communication
from midaGAN.utils.trackers.base import BaseTracker
from midaGAN.utils.trackers.utils import (process_visuals_for_logging,
                                          concat_batch_of_visuals_after_gather,
                                          convert_to_list_if_gather_did_not_occur)

import numpy as np


class ValTestTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logger

        self.metrics = []
        self.visuals = []

    def add_sample(self, visuals, metrics):
        """Parameters: # TODO: update this
        """
        metrics = {k: v for k, v in metrics.items() if v is not None}
        visuals = {k: v for k, v in visuals.items() if v is not None}

        # Gatther metrics and send to the process of rank 0
        metrics = communication.gather(metrics)
        metrics = convert_to_list_if_gather_did_not_occur(metrics)
        # Gather visuals from different processes to the rank 0 process
        visuals = communication.gather(visuals)
        visuals = concat_batch_of_visuals_after_gather(visuals)
        visuals = process_visuals_for_logging(self.conf,
                                              visuals,
                                              single_example=False,
                                              mid_slice_only=True)

        self.visuals.extend(visuals)
        self.metrics.extend(metrics)

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

        metrics_dict = {}
        # self.metrics containts a list of dicts with different metrics
        for metric in self.metrics:
            #  Each key in metric dict containts a list of values corresponding to
            #  each batch
            for metric_name, metric_list in metric.items():
                if metric_name in metrics_dict:
                    metrics_dict[metric_name].extend(metric_list)
                else:
                    metrics_dict[metric_name] = metric_list

        # Averages collected metrics from different iterations and batches
        averaged_metrics = {k: np.mean(v) for k, v in metrics_dict.items()}

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
