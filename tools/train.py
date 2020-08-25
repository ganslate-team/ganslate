import sys
import torch
from omegaconf import OmegaConf
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# --------- midaGAN imports ----------
try:
    import midaGAN
except ImportError:
    logger.warning("midaGAN not installed as a package, importing it from the local directory.")
    sys.path.append('./')
    import midaGAN

from midaGAN.utils import communication
from midaGAN.utils.environment import setup_logging
from midaGAN.trainer import Trainer
from midaGAN.conf import init_config
# -------------------------------------


def train():
    # experiment_dir = base_directory / run_name

    # if communication.get_local_rank() == 0:
        # Want to prevent multiple workers from trying to write a directory
        # This is required in the logging below
        # pass
        # experiment_dir.mkdir(parents=True, exist_ok=True)
    # communication.synchronize()  # Ensure folders are in place.

    # log_file = experiment_dir / f'log_{machine_rank}_{communication.get_local_rank()}.txt'
    

    cli = OmegaConf.from_cli()
    conf = init_config(cli.config)
    cli.pop("config")
    conf = OmegaConf.merge(conf, cli)
    
    
    trainer = Trainer(conf)


    # TODO: clean this mess
    log_file = Path(conf.logging.output_dir) / Path('log.txt')
    debug = False
    setup_logging(
        use_stdout=communication.get_local_rank() == 0 or debug,
        filename=log_file,
        log_level=('INFO' if not debug else 'DEBUG')
    )
    # TODO: all this should be in trainer.py
    # logger.info(f'Machine rank: {machine_rank}.')
    logger.info(f'Local rank: {communication.get_local_rank()}.')
    logger.info(f'Logging: {log_file}.')
    # logger.info(f'Saving to: {experiment_dir}.')
    # logger.info(f'Run name: {run_name}.')
    # logger.info(f'Config file: {cfg_filename}.')
    logger.info(f'Python version: {sys.version.strip()}.')
    logger.info(f'PyTorch version: {torch.__version__}.')  # noqa
    logger.info(f'CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()}.')
    logger.info(f'Configuration: {conf.pretty()}.')


    trainer.train()


if __name__ == '__main__':
    train()
