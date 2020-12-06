import sys
import logging
logger = logging.getLogger(__name__)

# --------- midaGAN imports ----------
try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.trainer import Trainer
from midaGAN.conf.builders import build_training_conf
from midaGAN.utils import communication, environment
from midaGAN.data import build_loader
from omegaconf import OmegaConf
import wandb

def main():
    communication.init_distributed()  # inits distributed mode if ran with torch.distributed.launch

    conf = build_training_conf()

    environment.setup_logging_with_config(conf)
    
    # Load limited entries in the dataloader
    conf.gan.is_train = False 
    data_loader = build_loader(conf)

    project = conf.logging.wandb.project
    entity = conf.logging.wandb.entity
    config_dict = OmegaConf.to_container(conf, resolve=True)

    wandb.init(name='dataset_check', project=project, entity=entity, config=config_dict) #
    
    for idx in range(10):
        for i, data in enumerate(data_loader):
            print(f"Loading {i}/{len(data_loader)} @ {idx} Pass")

            # Check for n NCDHW
            if data['A'].ndim == 5:
                image_A = data['A'][0, 0, conf.dataset.patch_size[0]//2]
                image_B = data['B'][0, 0, conf.dataset.patch_size[0]//2]
            elif data['A'].ndim == 4:
                image_A = data['A'][0, 0]
                image_B = data['B'][0, 0]                
            else: 
                return NotImplementedError("Only 2D and 3D datasets are supported")

            log_dict = {

                "A":wandb.Image(image_A.cpu().detach().numpy()), 
                "B":wandb.Image(image_B.cpu().detach().numpy())
            }

            wandb.log(log_dict)

if __name__ == '__main__':
    main()
