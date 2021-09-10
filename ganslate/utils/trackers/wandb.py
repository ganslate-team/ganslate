from omegaconf import OmegaConf
import wandb
import os
import torch
import numpy as np
from ganslate.utils.trackers.utils import process_visuals_wandb_tensorboard


def torch_npy_to_python(x):
    if isinstance(x, torch.Tensor) or np.isscalar(x):
        return x.item()
    return x


class WandbTracker:

    def __init__(self, conf):
        project = conf[conf.mode].logging.wandb.project
        entity = conf[conf.mode].logging.wandb.entity
        conf_dict = OmegaConf.to_container(conf, resolve=True)
        run_dir = conf[conf.mode].output_dir

        if wandb.run is None:
            if conf[conf.mode].checkpointing.load_iter and conf[conf.mode].logging.wandb.id:
                # Source: https://docs.wandb.ai/library/resuming
                os.environ["WANDB_RESUME"] = "allow"
                os.environ["WANDB_RUN_ID"] = conf[conf.mode].logging.wandb.id

            wandb.init(project=project, entity=entity, config=conf_dict, dir=run_dir)

        if conf[conf.mode].logging.wandb.run:
            wandb.run.name = conf[conf.mode].logging.wandb.run

        self.image_window = None
        if conf[conf.mode].logging.image_window:
            self.image_window = conf[conf.mode].logging.image_window

    def log_iter(self,
                 iter_idx,
                 visuals,
                 mode,
                 learning_rates=None,
                 losses=None,
                 metrics=None):
        """"""
        log_dict = {}

        # Learning rates
        if learning_rates:
            for name, learning_rate in learning_rates.items():
                log_dict[f"Learning rate: {name}"] = learning_rate

        # Losses
        if losses:
            for name, loss in losses.items():
                log_dict[f"Loss: {name}"] = torch_npy_to_python(loss)

        # Metrics
        if metrics:
            for name, metric in metrics.items():
                log_dict[f"Metric: {name} ({mode})"] = torch_npy_to_python(metric)

        normal_visuals = process_visuals_wandb_tensorboard(visuals,
                                                           image_window=None,
                                                           is_wandb=True)
        log_dict[f"Images ({mode})"] = normal_visuals

        if self.image_window:
            windowed_visuals = process_visuals_wandb_tensorboard(visuals,
                                                                 self.image_window,
                                                                 is_wandb=True)
            log_dict[f"Windowed images ({mode})"] = windowed_visuals
        wandb.log(log_dict, step=iter_idx)
