import torch
from data import build_loader
from nn import build_model
from utils.distributed import init_distributed, communication

from utils.logging.experiment_tracker import ExperimentTracker

from omegaconf import OmegaConf
from conf import init_config

class Trainer:
    def __init__(self, conf):
        if conf.distributed:
            init_distributed()

        self.tracker = ExperimentTracker(conf)
        self.data_loader = build_loader(conf)
        self.model = build_model(conf)

        self.iters = range(conf.continue_iter, 1 + conf.n_iters + conf.n_iters_decay)
        self.iter_idx = 0
        self.checkpoint_freq = conf.logging.checkpoint_freq

    def train(self):
        if communication.is_main_process():
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
        self.tracker.close()

    def _do_iteration(self, data):
        self.model.set_input(data)
        self.model.optimize_parameters()

    def _perform_scheduler_step(self):
        self.model.update_learning_rate()  # perform a scheduler step # TODO: better to make decaying rate in checkpoints rather than per iter

    def _save_checkpoint(self):
        if communication.is_main_process():
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
