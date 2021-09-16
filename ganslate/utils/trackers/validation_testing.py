from loguru import logger
from pathlib import Path

import torch

from ganslate.utils import communication
from ganslate.utils.trackers.base import BaseTracker
from ganslate.utils.csv_saver import Saver
from ganslate.utils.trackers.utils import (process_visuals_for_logging,
                                          concat_batch_of_visuals_after_gather,
                                          convert_to_list_if_gather_did_not_occur)

import numpy as np

class ValTestTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logger

        # Save to csv
        if getattr(self.conf[self.conf.mode].metrics, "save_to_csv", False):
            self.saver = Saver()
        else:
            self.saver = None

        self.metrics = []
        self.visuals = []

    def add_sample(self, visuals, metrics):
        """
        """
        def parse_metrics_to_buffer(metrics):
            metrics = {k: v for k, v in metrics.items() if v is not None}
            # Gather metrics and send to the process of rank 0
            metrics = communication.gather(metrics)
            metrics = convert_to_list_if_gather_did_not_occur(metrics)
            self.metrics.extend(metrics)
        
        def parse_visuals_to_buffer(visuals):
            visuals = {k: v for k, v in visuals.items() if v is not None}
            # Gather visuals from different processes to the rank 0 process
            visuals = communication.gather(visuals)
            visuals = concat_batch_of_visuals_after_gather(visuals)
            visuals = process_visuals_for_logging(self.conf,
                                                visuals,
                                                single_example=False,
                                                mid_slice_only=True)
            self.visuals.extend(visuals)

        parse_visuals_to_buffer(visuals)
        parse_metrics_to_buffer(metrics)

    def log_samples(self, iter_idx, dataset_name):
        """
        """
        def save_metrics(metrics_dict):
            # Save individual metrics if enabled
            if self.saver:
                n_samples = len(list(metrics_dict.values())[0])
                for index in range(n_samples):
                    row = {}
                    for metric_name, metric_list in metrics_dict.items():
                        row[metric_name] = metric_list[index]
                    self.saver.add(row)

                filepath = Path(self.output_dir) / "metrics.csv"
                self.saver.write(filepath)


        def get_metrics():
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
 
            save_metrics(metrics_dict)
            # Averages collected metrics from different iterations and batches
            return {k: np.mean(v) for k, v in metrics_dict.items()}

        def log_message():
            message = '\n' + 20 * '-' + f" ({self.conf.mode.capitalize()}"
            if iter_idx is not None:
                message += f" at iter {iter_idx}"
            if dataset_name is not None:
                message += f" for dataset '{dataset_name}'"
            message += ') ' + 20 * '-' + '\n'

            for name, metric in metrics.items():
                name = f'{dataset_name}_{name}' if dataset_name is not None else str(name)
                message += f"{name}: {metric:.3f} "
            self.logger.info(message)

        def log_visuals():
            for visuals_idx, visuals in enumerate(self.visuals):
                name = ""
                if dataset_name is not None:
                    name += f"{dataset_name}/"
                if iter_idx is not None:
                    name += f"{iter_idx}"
                    # When val, put images in a dir for the iter at which it is validating.
                    # When testing, there aren't multiple iters, so it isn't necessary.
                    name += "/" if self.conf.mode == "val" else "_"
                name += f"{visuals_idx}"
                
                self._save_image(visuals, name)

        def clear_buffers():
            # Clear stored buffer after logging the results
            self.metrics = []
            self.visuals = []

            
        metrics = get_metrics()
        log_message()
        log_visuals()

        mode = self.conf.mode
        if dataset_name is not None:
            mode = f"{mode}_{dataset_name}"

        if self.wandb:
            self.wandb.log_iter(iter_idx=iter_idx,
                                visuals=self.visuals,
                                mode=mode,
                                metrics=metrics)

        if self.tensorboard:
            self.tensorboard.log_iter(iter_idx=iter_idx,
                                      learning_rates=None,
                                      losses=None,
                                      visuals=self.visuals,
                                      mode=mode,
                                      metrics=metrics)

        clear_buffers()
