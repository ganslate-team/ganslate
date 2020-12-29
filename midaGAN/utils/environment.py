# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import os
import sys
from typing import Optional
from os import PathLike
from pathlib import Path

import numpy as np
import random
import cv2

import torch
from omegaconf import OmegaConf
import SimpleITK as sitk
from midaGAN.utils import io, communication

logger = logging.getLogger(__name__)


def setup_logging_with_config(conf, debug=False):
    use_stdout = communication.get_local_rank() == 0 or debug
    log_level = 'INFO' if not debug else 'DEBUG'

    if conf.is_train:
        output_dir = Path(conf.logging.checkpoint_dir).resolve()
        saving_to_message = f'Saving checkpoints, logs and config to: {output_dir}'
        filename = Path(output_dir) / 'training_log.txt'
    else:
        output_dir = Path(conf.logging.inference_dir).resolve()
        saving_to_message = f'Saving inference outputs, logs and config to: {output_dir}'
        filename = Path(output_dir) / 'inference_log.txt'

    io.mkdirs(output_dir)

    setup_logging(use_stdout, filename, log_level=log_level)

    logger.info(f'Configuration:\n{OmegaConf.to_yaml(conf)}')
    logger.info(saving_to_message)
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

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

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


def set_seed(seed=0):
    # Inspired also from: https://stackoverflow.com/a/57417097
    logger.info(f"Reproducible mode ON with seed : {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def threading_setup():
    """
    Sets max threads for SimpleITK and Opencv.
    For numpy etc. set OMP_NUM_THREADS=1 as an env var while 
    running the training script 
    E.g 
    OMP_NUM_THREADS=1 python tools/train.py ...
    """
    logger.warning("""
    Max threads for SimpleITK and Opencv set to 1
    For numpy etc. set OMP_NUM_THREADS=1 as an env var while 
    running the training script 
    E.g 
    OMP_NUM_THREADS=1 python tools/train.py ...
    """)
    MAX_THREADS = 1
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(MAX_THREADS)
    cv2.setNumThreads(MAX_THREADS)
