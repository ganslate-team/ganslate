import os
import logging
import torch

from midaGAN.data import build_loader
from midaGAN.nn.gans import build_gan

from midaGAN.utils import communication, environment
from midaGAN.utils.trackers.training_tracker import TrainingTracker
from midaGAN.utils.summary import gan_summary

# Imports for evaluation.
<<<<<<< HEAD
from midaGAN.conf.builders import build_eval_conf
from midaGAN.inferer import Inferer as Evaluator
from midaGAN.nn.metrics.eval_metrics import EvaluationMetrics
=======
from midaGAN.evaluator import Evaluator
>>>>>>> e12b23b0ddd49bb52b6c7ca61b1a58a1204be971

class Trainer():
    def __init__(self, conf):
        self.logger = logging.getLogger(type(self).__name__)
        self.conf = conf

        torch.backends.cudnn.benchmark = True # https://stackoverflow.com/a/58965640
        
        # Set reproducibility parameters (random numbers and cudnn backend)
        if self.conf.seed:
            environment.set_seed(self.conf.seed)

        self.tracker = TrainingTracker(self.conf)

        self.data_loader = build_loader(self.conf)

        self.model = build_gan(self.conf)

        # Evaluation configuration and evaluation dataloader specified. 
        self._init_evaluation(self.conf)

        start_iter = 1 if not self.conf.load_checkpoint else self.conf.load_checkpoint.count_start_iter
        end_iter = 1 + self.conf.n_iters + self.conf.n_iters_decay
        self.iters = range(start_iter, end_iter)
        self.iter_idx = 0

        self.checkpoint_freq = self.conf.logging.checkpoint_freq
        self.eval_freq = self.conf.eval.eval_freq

    def run(self):
        # self.logger.info(gan_summary(self.model, self.data_loader)) TODO: breaks 3D training with num_workers>0
        self.logger.info('Training started.')

        self.tracker.start_dataloading_timer()
        for i, data in zip(self.iters, self.data_loader):
            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()
            self._set_iter_idx(i)
            
            self._do_iteration(data)
            self.tracker.end_computation_timer()
            
            learning_rates, losses, visuals, metrics = self.model.get_loggable_data()
            self.tracker.log_iter(learning_rates, losses, visuals, metrics)

            self._save_checkpoint()
            self._perform_scheduler_step()
            
            self.evaluate()

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


    def _init_evaluation(self, conf):
        """
        Intitialize evaluation parameters from training conf.
        """
<<<<<<< HEAD

        # Eval conf is built from training conf + override from inference conf
        self.eval_conf = build_eval_conf(conf)

        if self.eval_conf:
            # Inferer is imported as Evaluator
            self.evaluator = Evaluator(self.eval_conf)

            # Evaluator model is set to training model. For regular 
            # inference this would  have been loaded from a checkpoint
            self.evaluator.model = self.model
            self.eval_metrics = EvaluationMetrics(self.eval_conf)


    def evaluate(self):
        if self.eval_conf and self.iter_idx % self.eval_freq == 0:
            # Disable training mode for inference!
            self.model.is_train = False

            self.logger.info(f"Evaluation started, running with {self.eval_conf.samples} samples")
            for i, data in zip(range(self.eval_conf.samples + 1), self.evaluator.data_loader):
                self.logger.info(f"Running eval on sample: {i}")

                A = data['A'].to(self.model.device)
                fake_B = self.evaluator.infer(A)
                rec_A = self.evaluator.infer(fake_B, infer_fn='infer_backward')

                # Full volume logging! This uploads slice-by-slice
                # inference over entire volume. 
                metrics = {}

                if self.eval_conf.metrics.ssim:
                    metrics.update({
                        'SSIM_A->B->A': self.eval_metrics.SSIM(A, rec_A)
                    })

                self.tracker.log_iter({}, {}, {
                    'real_A': A,
                    'fake_B': fake_B,
                    'rec_A': rec_A
                }, metrics)

            # Renable training mode. Does this affect anything dynamically? Check
            self.model.is_train = True

=======
        # Eval conf is built from training conf
        self.evaluator = Evaluator(conf)
        self.evaluator.set_model(self.model)

    def evaluate(self):
        if self.iter_idx % self.conf.evaluation.freq == 0:
            self.evaluator.run()
>>>>>>> e12b23b0ddd49bb52b6c7ca61b1a58a1204be971

    def _set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx
        self.tracker.set_iter_idx(iter_idx)
        self.evaluator.set_trainer_idx(iter_idx)


<<<<<<< HEAD

=======
>>>>>>> e12b23b0ddd49bb52b6c7ca61b1a58a1204be971
