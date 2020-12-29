import wandb
from omegaconf import OmegaConf


class WandbTracker:

    def __init__(self, conf):
        project = conf.logging.wandb.project
        entity = conf.logging.wandb.entity

        config_dict = OmegaConf.to_container(conf, resolve=True)

        wandb.init(project=project, entity=entity,
                   config=config_dict)  # TODO: project and organization from conf

        if conf.logging.wandb.run:
            wandb.run.name = conf.logging.wandb.run

        self.image_filter = None
        if conf.logging.wandb.image_filter:
            self.image_filter = [
                conf.logging.wandb.image_filter.min, conf.logging.wandb.image_filter.max
            ]

    def log_iter(self, iter_idx, learning_rates, losses, visuals, metrics, batch=None):
        """TODO"""
        log_dict = {}

        # Iteration idx
        log_dict['iter_idx'] = iter_idx

        # Learning rates
        if learning_rates:
            log_dict['lr_G'] = learning_rates['lr_G']
            log_dict['lr_D'] = learning_rates['lr_D']

        # Losses
        for name, loss in losses.items():
            log_dict[f"loss_{name}"] = loss

        # Metrics
        for name, metric in metrics.items():
            log_dict[name] = metric

        # Image
        name, image = visuals['name'], visuals['image']
        image = image.permute(1, 2, 0)  # CxHxW -> HxWxC
        log_dict[name] = [wandb.Image(image.cpu().detach().numpy())]

        if self.image_filter:
            filter_min, filter_max = self.image_filter
            image = image.clamp(filter_min, filter_max)
            image = (image - filter_min) / filter_max - filter_min
            log_dict[f"{name}_filtered"] = [wandb.Image(image.cpu().detach().numpy())]

        if batch:
            log_dict["batch"] = batch

        wandb.log(log_dict)
