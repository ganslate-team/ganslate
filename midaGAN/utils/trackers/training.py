from loguru import logger
from pathlib import Path

from midaGAN.utils import communication
from midaGAN.utils.trackers.base import BaseTracker
from midaGAN.utils.trackers.utils import process_visuals_for_logging


class TrainingTracker(BaseTracker):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logger
        self.log_freq = conf.train.logging.freq

    def log_iter(self, learning_rates, losses, visuals, metrics):
        """Parameters: # TODO: update this
            iters (int) -- current training iteration
            losses (tuple/list) -- training losses
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if self.iter_idx % self.log_freq != 0:
            return

        def parse_visuals(visuals):
            # Note: Gather not necessary as in val/test, enough to log one example when training.
            visuals = {k: v for k, v in visuals.items() if v is not None}
            visuals = process_visuals_for_logging(self.conf, visuals, single_example=True)
            # `single_example=True` returns a single example from the batch, selecting it
            return visuals[0]

        def parse_losses(losses):
            losses = {k: v for k, v in losses.items() if v is not None}
            # Reduce losses (avg) and send to the process of rank 0
            losses = communication.reduce(losses, average=True, all_reduce=False)
            return losses

        def parse_metrics(metrics):
            metrics = {k: v for k, v in metrics.items() if v is not None}
            # Training metrics are optional
            if metrics:
                # Reduce metrics (avg) and send to the process of rank 0
                metrics = communication.reduce(metrics, average=True, all_reduce=False)
            return metrics

        def log_message():
            lr_G, lr_D = learning_rates["lr_G"], learning_rates["lr_D"]
            message = '\n' + 20 * '-' + ' '
            message += (f"(iter: {self.iter_idx} | comp: {self.t_comp:.3f}, "
                        f"data: {self.t_data:.3f} | lr_G: {lr_G:.7f}, lr_D = {lr_D:.7f})")
            message += ' ' + 20 * '-' + '\n'

            for k, v in losses.items():
                message += f"{k}: {v:.3f} "
            self.logger.info(message)

        def log_visuals():
            self._save_image(visuals, self.iter_idx)

        visuals = parse_visuals(visuals)
        losses = parse_losses(losses)
        metrics = parse_metrics(metrics)

        log_message()
        log_visuals()

        if self.wandb:
            self.wandb.log_iter(iter_idx=self.iter_idx,
                                visuals=visuals,
                                mode=self.conf.mode,
                                learning_rates=learning_rates,
                                losses=losses,
                                metrics=metrics)

        if self.tensorboard:
            raise NotImplementedError("Tensorboard tracking not implemented")
            # self.tensorboard.log_iter(self.iter_idx, learning_rates, losses, visuals, metrics)

