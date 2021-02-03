from abc import abstractmethod
from pathlib import Path

from midaGAN.data import build_loader
from midaGAN.data.utils import decollate
from midaGAN.engines.base import BaseEngineWithInference
from midaGAN.nn.gans import build_gan
from midaGAN.nn.metrics.eval_metrics import EvaluationMetrics
from midaGAN.utils import environment
from midaGAN.utils.trackers.evaluation import EvaluationTracker


class BaseEvaluator(BaseEngineWithInference):
    def __init__(self, conf):
        super().__init__(conf)
        self.output_dir = Path(conf.train.logging.checkpoint_dir) / self.conf.mode

        self.data_loader = build_loader(self.conf)
        self.tracker = EvaluationTracker(self.conf)
        self.metrics = EvaluationMetrics(self.conf)
        self.trainer_idx = 0

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
        pred = visuals["fake_B"]
        target = visuals["B"]

        # Check if dataset has `denormalize` method defined,
        # if not, compute the metrics in [0, 1] space
        if denormalize := getattr(self.data_loader.dataset, "denormalize", False):
            pred = denormalize(pred)
            target = denormalize(target)

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
        # by inspecting if the model's `infer()` method contains `cycle` argument.
        # TODO: Improve placement
        if self.conf[self.conf.mode].metrics.cycle_metrics:
            assert 'cycle' in self.model.infer.__code__.co_varnames, \
            "If cycle metrics are enabled, please define behavior of inference"\
            "with a `cycle` flag in the model's `infer()` method"
            rec_A = self.infer(visuals["fake_B"], cycle='B')
            metrics.update(self.metrics.get_cycle_metrics(rec_A, visuals["A"]))

        return metrics


class Validator(BaseEvaluator):
    def __init__(self, conf, model):
        super().__init__(conf)
        self.model = model

    def _set_mode(self):
        self.conf.mode = 'val'


class Tester(BaseEvaluator):
    def __init__(self, conf):
        super().__init__(conf)
        environment.setup_logging_with_config(self.conf)
        self.model = build_gan(self.conf)

    def _set_mode(self):
        self.conf.mode = 'test'

    def _override_conf(self):
        self.conf.train.load_checkpoint = self.conf.test.checkpoint_iter
