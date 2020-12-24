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
        self.eval_enabled = conf.evaluation is not None
        # Load evaluation configuration from training configuration!
        if self.eval_enabled:
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

        if self.eval_enabled:
            self.eval_iter_idx = 1

            self.model.is_train = False
            self.logger.info(f"Evaluation started, running with {self.conf.samples} samples")
            for i, data in zip(range(self.conf.samples + 1), self.data_loader):
                
                metadata = decollate(data['metadata'])
                predicted = self.infer(data['A'])
                metrics = self.metrics.get_metric_dict(predicted, data['B'])
                
                self.tracker.log_sample(self.trainer_idx, self.eval_iter_idx, {"A": data['A'].to(self.model.device), 
                                                                                "B": data['B'].to(self.model.device),
                                                                                "fake_B": predicted
                                                                                }, metrics)
                
                self.data_loader.dataset.save(predicted, metadata, inference_dir / f"{self.trainer_idx}_{self.eval_iter_idx}")
                
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

    