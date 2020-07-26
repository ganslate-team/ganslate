
import wandb

class WandbTracker:
    def __init__(self, conf):
        wandb.init(project="my-project", config=dict(conf)) # TODO: project and organization from conf

    def log_iter(self, iter_idx, learning_rates, losses, visuals):
        """TODO"""
        log_dict = {}

        # Iteration idx
        log_dict['iter_idx'] = iter_idx

        # Learning rates
        log_dict['lr_G'] = learning_rates['lr_G']
        log_dict['lr_D'] = learning_rates['lr_D']
        
        # Losses
        for name, loss in losses.items():
            log_dict['loss_%s' % name] = loss
        
        # Image
        name, image = visuals['name'], visuals['image']
        image = image.permute(1,2,0) # CxHxW -> HxWxC
        log_dict[name] = [wandb.Image(image.cpu().detach().numpy())]

        wandb.log(log_dict)
