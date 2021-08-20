from omegaconf import OmegaConf
import wandb
import os
import torch
import numpy as np


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

        self.image_filter = None
        if conf[conf.mode].logging.wandb.image_filter:
            self.image_filter = conf[conf.mode].logging.wandb.image_filter

    def log_iter(self,
                 iter_idx,
                 visuals,
                 mode='train',
                 learning_rates=None,
                 losses=None,
                 metrics=None):
        """"""
        log_dict = {}

        # Learning rates
        if learning_rates:
            log_dict['lr_G'] = learning_rates['lr_G']
            log_dict['lr_D'] = learning_rates['lr_D']

        # Losses
        if losses:
            for name, loss in losses.items():
                log_dict[f"loss_{name}"] = torch_npy_to_python(loss)

        # Metrics
        if metrics:
            for name, metric in metrics.items():
                log_dict[f"metric_{mode}_{name}"] = torch_npy_to_python(metric)

        log_dict[f"{mode} Images"] = self.create_wandb_images(visuals)

        if self.image_filter:
            log_dict[f"{mode} Windowed Images"] = self.create_wandb_images(
                visuals, image_threshold=self.image_filter)

        wandb.log(log_dict, step=iter_idx)

    def create_wandb_images(self, visuals, image_threshold=None):
        """
        Create wandb images from visuals
        """
        # Check if visuals is a list of images and create a list of wandb.Image's
        if isinstance(visuals, list):
            wandb_images = []
            for idx, visual in enumerate(visuals):
                # Add sample index to visual name to identify it.
                if image_threshold is not None:
                    visual['name'] = f"{idx}_{visual['name']}"

                visual = self._wandb_image_from_visual(visual, image_threshold)
                wandb_images.append(visual)
            return wandb_images

        # If visual is an image then a single wandb.Image is created
        return self._wandb_image_from_visual(visuals, image_threshold=image_threshold)

    def _wandb_image_from_visual(self, visual, image_threshold=None):
        """
        Wandb Image Reference:
        https://docs.wandb.ai/library/log#images-and-overlays
        """
        name, image = visual['name'], visual['image']
        # CxHxW -> HxWxC
        image = image.permute(1, 2, 0)

        # Check if a threshold is defined while creating the wandb image.
        if image_threshold:
            image = image.clamp(image_threshold[0], image_threshold[1])
            image = (image - image_threshold[0]) / image_threshold[1] - image_threshold[0]

        return wandb.Image(image.cpu().detach().numpy(), caption=name)
