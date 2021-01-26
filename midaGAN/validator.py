import logging
from pathlib import Path

from monai.inferers import SlidingWindowInferer

from midaGAN.configs.utils import builders
from midaGAN.data import build_loader
from midaGAN.data.utils import decollate
from midaGAN.nn.metrics.eval_metrics import EvaluationMetrics
from midaGAN.utils.trackers.val_tracker import ValidationTracker


class Validator():

    def __init__(self, conf):
        self.logger = logging.getLogger(type(self).__name__)
        self.enabled = conf.validation is not None
        # Load validation configuration from training configuration
        if self.enabled:
            self.conf = builders.build_val_conf(conf)
            self.logger.info(f"Validation configuration \n {self.conf.pretty()}")

            self.data_loader = build_loader(self.conf)
            self.tracker = ValidationTracker(self.conf)
            self.sliding_window_inferer = self._init_sliding_window_inferer()
            self.metrics = EvaluationMetrics(self.conf)

        self.trainer_idx = 0

    def set_trainer_idx(self, idx):
        self.trainer_idx = idx

    def set_model(self, model):
        self.model = model

    def run(self):
        inference_dir = Path(self.conf.logging.inference_dir) / "val_nrrds"

        if self.enabled:
            self.logger.info(f"Running validation with {len(self.data_loader.dataset)} samples")

            for data in self.data_loader:
                # Move elements from data that are visuals
                visuals = {
                    "A": data['A'].to(self.model.device),
                    "fake_B": self.infer(data['A']),
                    "B": data['B'].to(self.model.device)
                }

                # Add masks to visuals if they are provided
                if "masks" in data:
                    visuals.update({"masks": data["masks"]})

                metrics = self.calculate_metrics(visuals)
                self.tracker.add_sample(visuals, metrics)
                metadata = decollate(data['metadata'])
                self.data_loader.dataset.save(visuals['fake_B'], metadata,
                                              inference_dir / f"{self.trainer_idx}")

            self.tracker.push_samples(self.trainer_idx)

    def infer(self, data, *args, **kwargs):
        data = data.to(self.model.device)
        # Sliding window (i.e. patch-wise) inference
        if self.sliding_window_inferer:
            return self.sliding_window_inferer(data, lambda x: self.model.infer(x, *args, **kwargs))
        return self.model.infer(data)

    def _init_sliding_window_inferer(self):
        if not self.conf.sliding_window:
            return

        return SlidingWindowInferer(roi_size=self.conf.sliding_window.window_size,
                                    sw_batch_size=self.conf.sliding_window.batch_size,
                                    overlap=self.conf.sliding_window.overlap,
                                    mode=self.conf.sliding_window.mode,
                                    cval=-1)


    def calculate_metrics(self, visuals):
        # Input to Translated comparison metrics
        pred = visuals["fake_B"]
        target = visuals["A"]
        # Check if dataset has scale_to_hu method defined,
        # if not, compute the metrics in [0, 1] space
        if hasattr(self.data_loader.dataset, "scale_to_hu"):
            pred = self.data_loader.dataset.scale_to_hu(pred)
            target = self.data_loader.dataset.scale_to_hu(target)
        metrics = self.metrics.get_metrics(pred, target)
        
        # Check if any masks are defined and calculate mask specific metrics
        if "masks" in visuals:
            for label, mask in visuals["masks"].items():
                masked_pred = pred * mask.to(pred.device)
                masked_target = target * mask.to(target.device)
                masked_metrics = self.metrics.get_metrics(masked_pred, masked_target, suffix=label)
                metrics.update(masked_metrics)

            # Remove masks after calculating metrics - these dont need to be logged
            visuals.pop("masks")

        # Check if cycle metrics are enabled and if the model is cyclic
        # Cyclic check for model will be done through definition of 
        # cycle key in infer method.TODO: Improve placement
        if self.conf.metrics.cycle_metrics:
            assert 'cycle' in self.model.infer.__code__.co_varnames, \
            "If cycle metrics are enabled, please define behavior of inference"\
            "with a cycle flag"
            rec_A = self.infer(visuals["fake_B"], cycle='B')
            metrics.update(self.metrics.get_cycle_metrics(rec_A, visuals["A"])) 
    
        return metrics

    def is_enabled(self):
        return self.enabled
