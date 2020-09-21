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
# -------------------------------------
    # TODO: this might come in handy later
    # if communication.get_local_rank() == 0:
        # Want to prevent multiple workers from trying to write a directory
        # This is required in the logging below
        # pass
        # experiment_dir.mkdir(parents=True, exist_ok=True)
    # communication.synchronize()  # Ensure folders are in place.
    # log_file = experiment_dir / f'log_{machine_rank}_{communication.get_local_rank()}.txt'
    

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
