
import os
import logging

from midaGAN.data import build_loader
from midaGAN.nn.gans import build_gan

from midaGAN.utils import communication, environment
from midaGAN.utils.trackers.training_tracker import TrainingTracker
from midaGAN.utils.summary import gan_summary

class Trainer():
    def __init__(self, conf):
        self.logger = logging.getLogger(type(self).__name__)
        self.conf = conf


        # Set reproducibility parameters
        environment.set_seed(self.conf.seed)

        self.tracker = TrainingTracker(self.conf)

        self.data_loader = build_loader(self.conf)
        self.model = build_gan(self.conf)

        start_iter = 1 if not self.conf.load_checkpoint else self.conf.load_checkpoint.count_start_iter
        end_iter = 1 + self.conf.n_iters + self.conf.n_iters_decay
        self.iters = range(start_iter, end_iter)
        self.iter_idx = 0
        self.checkpoint_freq = self.conf.logging.checkpoint_freq

    def run(self):
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
        # TODO: save on cancel
        if communication.get_local_rank() == 0:
            if self.iter_idx % self.checkpoint_freq == 0:
                self.logger.info(f'Saving the model after {self.iter_idx} iterations.')
                self.model.save_checkpoint(self.iter_idx)

    def _set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx
        self.tracker.set_iter_idx(iter_idx)

    