# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import os
import sys
from typing import Optional
from os import PathLike
from pathlib import Path

import torch
from omegaconf import OmegaConf

from midaGAN.utils import io, communication


logger = logging.getLogger(__name__)

def setup_training_logging(conf, debug=False):
    use_stdout = communication.get_local_rank() == 0 or debug
    log_level = 'INFO' if not debug else 'DEBUG'

    checkpoint_dir = conf.logging.checkpoint_dir
    io.mkdirs(checkpoint_dir)
    filename = Path(checkpoint_dir) / 'log.txt'

    setup_logging(use_stdout, filename, log_level=log_level)

    logger.info(f'Configuration: {OmegaConf.to_yaml(conf)}')
    logger.info(f'Saving checkpoints, logs and config to: {checkpoint_dir}')
    logger.info(f'Python version: {sys.version.strip()}')
    logger.info(f'PyTorch version: {torch.__version__}')  # noqa
    logger.info(f'CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()}')
    
    # These two useful if we decide to keep logs of all processes
    #logger.info(f'Machine rank: {communication.get_rank()}.')  
    #logger.info(f'Local rank: {communication.get_local_rank()}.') 

    # -------------------------------------
    # TODO: this might come in handy later
    # if communication.get_local_rank() == 0:
        # Want to prevent multiple workers from trying to write a directory
        # This is required in the logging below
        # pass
        # experiment_dir.mkdir(parents=True, exist_ok=True)
    # communication.synchronize()  # Ensure folders are in place.
    # log_file = experiment_dir / f'log_{machine_rank}_{communication.get_local_rank()}.txt'


def setup_logging(use_stdout: Optional[bool] = True,
                  filename: Optional[PathLike] = None,
                  log_level: Optional[str] = 'INFO') -> None:
    """
    Setup logging for DIRECT.

    Parameters
    ----------
    use_stdout : bool
        Write output to standard out.
    filename : PathLike
        Filename to write log to.
    log_level : str
        Logging level as in the `python.logging` library.

    Returns
    -------
    None
    """
    if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'EXCEPTION']:
        raise ValueError(f'Unexpected log level got {log_level}.')

    logging.captureWarnings(True)
    log_level = getattr(logging, log_level)

    root = logging.getLogger('')
    root.setLevel(log_level)

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )

    if use_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
