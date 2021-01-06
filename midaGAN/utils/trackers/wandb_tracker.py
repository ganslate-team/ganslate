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


    def log_iter(self, iter_idx, learning_rates, losses, visuals, metrics, mode='train'):
        """TODO"""
        mode = mode.capitalize()
        log_dict = {}

        # Learning rates
        if learning_rates:
            log_dict['lr_G'] = learning_rates['lr_G']
            log_dict['lr_D'] = learning_rates['lr_D']

        # Losses
        for name, loss in losses.items():
            log_dict[f"loss_{name}"] = loss

        # Metrics
        for name, metric in metrics.items():
            log_dict[f"{mode} {name}"] = metric

        log_dict[f"{mode} Images"] = self.create_wandb_images(visuals)

        if self.image_filter:
            log_dict[f"{mode} Windowed Images"] = self.create_wandb_images(visuals, image_threshold=self.image_filter)

        wandb.log(log_dict, step=iter_idx)


    def create_wandb_images(self, visuals, image_threshold=None):
        # Check if visuals is a list of images and create a list
        # of wandb.Image
        if isinstance(visuals, list):
            wandb_images = []
            for idx, visual in enumerate(visuals):
                # Add sample index to visual name to identify it.
                if image_threshold is not None:
                    visual['name'] = f"Sample: {idx} {visual['name']}"
                    
                wandb_images.append(self.wandb_image_from_visual(visual))

            return wandb_images

        # If visual is an image then a single wandb.Image is created
        else:
            return self.wandb_image_from_visual(visuals)


    def wandb_image_from_visual(self, visual, image_threshold=None):
        """
        Wandb Image Reference:
        https://docs.wandb.ai/library/log#images-and-overlays
        """
        name, image = visual['name'], visual['image']
        image = image.permute(1, 2, 0)  # CxHxW -> HxWxC

        # Check if a threshold is defined while creating the wandb image. 
        if image_threshold:
            image = image.clamp(image_threshold[0], image_threshold[1])
            image = (image - image_threshold[0]) / image_threshold[1] - image_threshold[0]

        return wandb.Image(image.cpu().detach().numpy(), caption=name)