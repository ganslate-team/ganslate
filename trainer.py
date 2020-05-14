import torch
from data import build_loader
from models import build_model
from util.distributed import init_distributed, multi_gpu #, comm

from experiment_tracker import ExperimentTracker

from omegaconf import OmegaConf
from conf.initializer import init_config


class Trainer:
    def __init__(self, conf):
        if conf.distributed:
            init_distributed()

        self.model = build_model(conf)
        self.data_loader = build_loader(conf)
        self.tracker = ExperimentTracker(conf)

        self.iters = range(conf.continue_iter, 1 + conf.n_iters + conf.n_iters_decay)
        self.iter_idx = 0
        self.checkpoint_freq = conf.logging.checkpoint_freq

    def train(self):
        print('Training started.')
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

    def _do_iteration(self, data):
        self.model.set_input(data)
        self.model.optimize_parameters()

        self.learning_rates = self.model.get_learning_rates()
        self.losses = self.model.get_current_losses()

    def _perform_scheduler_step(self):
        self.model.update_learning_rate()  # perform a scheduler step 

    def _save_checkpoint(self):
        if multi_gpu.is_main_process(): # if epoch % conf.logging.save_epoch_freq == 0:
            if self.iter_idx % self.checkpoint_freq == 0:
                print('Saving the model after %d iterations.' % (self.iter_idx))
                self.model.save_checkpoint(self.iter_idx) #('latest')

    def _set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx
        self.tracker.set_iter_idx(iter_idx)


def main():
    conf = init_config('./conf/experiment1.yaml')
    cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli)
    print(conf.pretty())

    trainer = Trainer(conf)
    trainer.train()
            

if __name__ == '__main__':
    main()
