
import os
import logging
import torch
from omegaconf import OmegaConf

from midaGAN.data import build_loader
from midaGAN.nn.gans import build_gan
from midaGAN.utils import communication
from midaGAN.utils.environment import setup_logging
from midaGAN.utils.logging.experiment_tracker import ExperimentTracker
from midaGAN.utils.summary import gan_summary
from midaGAN.conf import init_config

class Trainer():
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)
        self.conf = self._build_config()

        self.tracker = ExperimentTracker(self.conf)
        self.data_loader = build_loader(self.conf)
        self.model = build_gan(self.conf)

        self.iters = range(self.conf.continue_iter, 
                           1 + self.conf.n_iters + self.conf.n_iters_decay)
        self.iter_idx = 0
        self.checkpoint_freq = self.conf.logging.checkpoint_freq

    def run(self):
        if communication.is_main_process():
            self.logger.info(gan_summary(self.model, self.data_loader))
            self.logger.info('Training started.')

        self.tracker.start_dataloading_timer()
        for i, data in zip(self.iters, self.data_loader):
            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()
            self._set_iter_idx(i)
            
            self._do_iteration(data)
            self.tracker.end_computation_timer()
            
            learning_rates, losses, visuals = self.model.get_loggable_data()
            self.tracker.log_iter(learning_rates, losses, visuals)

            self._save_checkpoint()
            self._perform_scheduler_step()
            
            self.tracker.start_dataloading_timer()
        self.tracker.close()

    def _do_iteration(self, data):
        self.model.set_input(data)
        self.model.optimize_parameters()

    def _perform_scheduler_step(self):
        self.model.update_learning_rate()  # perform a scheduler step # TODO: better to make decaying rate in checkpoints rather than per iter

    def _save_checkpoint(self):
        if communication.is_main_process():
            if self.iter_idx % self.checkpoint_freq == 0:
                self.logger.info(f'Saving the model after {self.iter_idx} iterations.')
                self.model.save_checkpoint(self.iter_idx) #('latest')

    def _set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx
        self.tracker.set_iter_idx(iter_idx)

    def _build_config(self):
        cli = OmegaConf.from_cli()
        conf = init_config(cli.pop("config"))
        return OmegaConf.merge(conf, cli)