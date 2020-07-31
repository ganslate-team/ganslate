
import os
import torch

from midaGAN import midaGanBase # TODO: make nicer

from midaGAN.data import build_loader
from midaGAN.nn import build_model
from midaGAN.utils import communication
from midaGAN.utils.environment import setup_logging
from midaGAN.utils.logging.experiment_tracker import ExperimentTracker


class Trainer(midaGanBase):
    def __init__(self, conf):
        # Training ran with torch.distributed.launch if there is 'WORLD_SIZE' environment variable
        if os.environ.get('WORLD_SIZE', None):
            communication.init_distributed()

        self.tracker = ExperimentTracker(conf)
        self.data_loader = build_loader(conf)
        self.model = build_model(conf)

        self.iters = range(conf.continue_iter, 1 + conf.n_iters + conf.n_iters_decay)
        self.iter_idx = 0
        self.checkpoint_freq = conf.logging.checkpoint_freq

        super().__init__()

    def train(self):
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