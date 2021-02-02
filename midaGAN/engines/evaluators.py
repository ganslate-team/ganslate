import logging
from pathlib import Path
from abc import abstractmethod

from omegaconf import open_dict
from monai.inferers import SlidingWindowInferer

from midaGAN.engines.inferer import Inferer
from midaGAN.configs.utils import builders
from midaGAN.data import build_loader
from midaGAN.data.utils import decollate
from midaGAN.nn.metrics.eval_metrics import EvaluationMetrics
from midaGAN.utils.trackers.evaluation import EvaluationTracker


class BaseEvaluator(Inferer):
    def __init__(self, conf, model):
        super().__init__(conf, model)

        self.logger = logging.getLogger(type(self).__name__)
        self.output_dir = Path(conf.train.logging.checkpoint_dir) / self.conf.mode

        self.data_loader = build_loader(self.conf)
        self.tracker = EvaluationTracker(self.conf)
        self.metrics = EvaluationMetrics(self.conf)
        self.trainer_idx = 0

    @abstractmethod
    def _set_mode(self):
        self.conf.mode = ...

    def set_trainer_idx(self, idx):
        self.trainer_idx = idx

    def run(self):
        self.logger.info(f"{self.conf.mode} with {len(self.data_loader.dataset)} samples")

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
            self.data_loader.dataset.save(visuals['fake_B'],
                                          metadata,
                                          self.output_dir / "nrrds" / str(self.trainer_idx))

        self.tracker.push_samples(self.trainer_idx)

    def calculate_metrics(self, visuals):
        # Input to Translated comparison metrics
        pred = visuals["fake_B"]
        target = visuals["A"]

        # TODO: this thing below is very bad, remove asap and do it somehow differently
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
        # cycle key in infer method. 
        # TODO: Improve placement
        if self.conf[self.conf.mode].metrics.cycle_metrics:
            assert 'cycle' in self.model.infer.__code__.co_varnames, \
            "If cycle metrics are enabled, please define behavior of inference"\
            "with a cycle flag"
            rec_A = self.infer(visuals["fake_B"], cycle='B')
            metrics.update(self.metrics.get_cycle_metrics(rec_A, visuals["A"]))

        return metrics


class Validator(BaseEvaluator):
    def _set_mode(self):
        self.conf.mode = 'val'


class Tester(BaseEvaluator):
    def _set_mode(self):
        self.conf.mode = 'test'