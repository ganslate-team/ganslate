import os
import logging
from pathlib import Path

import torch
import numpy as np
from monai.inferers import SlidingWindowInferer

from midaGAN.data import build_loader
from midaGAN.nn.gans import build_gan
from midaGAN.utils import io
from midaGAN.utils.trackers.eval_tracker import EvalTracker
from midaGAN.nn.metrics.eval_metrics import EvaluationMetrics
from midaGAN.data.utils import decollate

from midaGAN.conf.builders import build_eval_conf


class Evaluator():
    def __init__(self, conf):
        self.logger = logging.getLogger(type(self).__name__)
        self.enabled = conf.evaluation is not None
        # Load evaluation configuration from training configuration!
        if self.enabled:
            self.conf = build_eval_conf(conf)
            self.logger.info(f"Evaluation configuration \n {self.conf.pretty()}")

            self.data_loader = build_loader(self.conf)
            self.tracker = EvalTracker(self.conf)
            self.sliding_window_inferer = self._init_sliding_window_inferer()
            self.metrics = EvaluationMetrics(self.conf)

        self.trainer_idx = 0

    def set_trainer_idx(self, idx):
        self.trainer_idx = idx

    def set_model(self, model):
        self.model = model

    def run(self):
        inference_dir = Path(self.conf.logging.inference_dir) / "eval_nrrds"

        if self.enabled:
            self.eval_iter_idx = 1

            self.model.is_train = False
            self.logger.info(f"Evaluation started, running with {self.conf.samples} samples")
            for i, data in zip(range(self.conf.samples + 1), self.data_loader):
                
                # Move elements from data that are visuals
                visuals = {
                    "A": data['A'].to(self.model.device),
                    "B": data['B'].to(self.model.device)
                }

                visuals['fake_B'] = self.infer(visuals['A'])
                metrics = self.calculate_metrics(visuals['fake_B'], visuals['B'])

                self.tracker.log_sample(self.trainer_idx, self.eval_iter_idx, visuals, metrics)
                metadata = decollate(data['metadata'])
                self.data_loader.dataset.save(visuals['fake_B'], metadata, inference_dir / f"{self.trainer_idx}_{self.eval_iter_idx}")
                
                self.eval_iter_idx += 1

            self.model.is_train = True
            

    def infer(self, data):
        data = data.to(self.model.device)
        # Sliding window (i.e. patch-wise) inference
        if self.sliding_window_inferer:
            return self.sliding_window_inferer(data, self.model.infer)
        else:
            return self.model.infer(data)


    def _init_sliding_window_inferer(self):
        if self.conf.sliding_window:
            return SlidingWindowInferer(roi_size=self.conf.sliding_window.window_size,
                                        sw_batch_size=self.conf.sliding_window.batch_size,
                                        overlap=self.conf.sliding_window.overlap,
                                        mode=self.conf.sliding_window.mode, cval=-1)
        else:
            return None

    def calculate_metrics(self, pred, target):
        # Check if dataset has scale_to_HU method defined, 
        # if not, compute the metrics in [0, 1] space
        if hasattr(self.data_loader.dataset, "scale_to_HU"):
            pred = self.data_loader.dataset.scale_to_HU(pred)
            target =  self.data_loader.dataset.scale_to_HU(target)

        metrics = self.metrics.get_metric_dict(pred, target)
        return metrics

    def is_enabled(self):
        return self.enabled
    