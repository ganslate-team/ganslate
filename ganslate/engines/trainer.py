from loguru import logger

import torch
from ganslate.engines.base import BaseEngine
from ganslate.engines.validator_tester import Validator
from ganslate.utils import communication, environment
from ganslate.utils.builders import build_gan, build_loader
from ganslate.utils.trackers.training import TrainingTracker


class Trainer(BaseEngine):

    def __init__(self, conf):
        super().__init__(conf)
        environment.setup_logging_with_config(conf)

        # https://stackoverflow.com/a/58965640
        torch.backends.cudnn.benchmark = True

        # Set reproducibility parameters (random numbers and cudnn backend)
        if self.conf.train.seed:
            environment.set_seed(self.conf.train.seed)

        self.tracker = TrainingTracker(self.conf)

        self.data_loader = build_loader(self.conf)

        self.model = build_gan(self.conf)

        # Validation configuration and validation dataloader specified.
        self.validator = self._init_validator()

        start_iter = 1
        if self.conf.train.checkpointing.load_iter:
            start_iter += self.conf.train.checkpointing.load_iter

        end_iter = 1 + self.conf.train.n_iters + self.conf.train.n_iters_decay
        assert start_iter < end_iter, \
            "If continuing, define the `n_iters` relative to the loaded iteration."

        self.iters = range(start_iter, end_iter)
        self.iter_idx = 0

    def _set_mode(self):
        self.conf.mode = "train"

    def run(self):
        #TODO: breaks 3D training with num_workers>0. Create a temp dataloader from the dataset in gan_summary?
        # self.logger.info(gan_summary(self.model, self.data_loader))

        self.logger.info('Training started.')

        self.tracker.start_dataloading_timer()
        for i, data in zip(self.iters, self.data_loader):
            self._set_iter_idx(i)
            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()

            self._run_iteration(data)
            self.tracker.end_computation_timer()

            learning_rates, losses, visuals, metrics = self.model.get_loggable_data()
            self.tracker.log_iter(learning_rates, losses, visuals, metrics)

            self._save_checkpoint()
            self._perform_scheduler_step()

            self._run_validation()

            self.tracker.start_dataloading_timer()

        # Close tensorboard trackers
        self.tracker.close()
        if self.validator:
            self.validator.tracker.close()

    def _run_iteration(self, data):
        self.model.set_input(data)
        self.model.optimize_parameters()

    def _perform_scheduler_step(self):
        # perform a scheduler step
        self.model.update_learning_rate()

    def _save_checkpoint(self):
        # TODO: save on cancel
        if communication.get_rank() == 0:
            checkpoint_freq = self.conf.train.checkpointing.freq
            checkpoint_after = self.conf.train.checkpointing.start_after
            if self.iter_idx % checkpoint_freq == 0 and self.iter_idx >= checkpoint_after:
                self.logger.info(f'Saving the model after {self.iter_idx} iterations.')
                self.model.save_checkpoint(self.iter_idx)

    def _init_validator(self):
        """
        Intitialize validation parameters from training conf
        """
        # Validation conf is built from training conf
        if not self.conf.val:
            return
        return Validator(self.conf, self.model)

    def _run_validation(self):
        if self.validator:
            val_freq = self.conf.val.freq
            val_after = self.conf.val.start_after
            if self.iter_idx % val_freq == 0 and self.iter_idx >= val_after:
                self.validator.run(current_idx=self.iter_idx)

    def _set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx
        self.tracker.set_iter_idx(iter_idx)
