
import logging
import time

from util.distributed import comm, multi_gpu
from util.file_utils import mkdirs

class ExperimentTracker:
    def __init__(self, conf):
        self.logger = logging.getLogger(type(self).__name__)
        self.output_dir = conf.logging.output_dir
        mkdirs(self.output_dir)
        self.batch_size = conf.batch_size

        self.total_steps = 0
        self.iter_idx = 0
        self.iter_end_time = None
        self.iter_start_time = None
        self.t_data = None
        self.t_comp = None

    def set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx

    def start_computation_timer(self):
        self.iter_start_time = time.time()

    def start_dataloading_timer(self):
        self.iter_end_time = time.time()
    
    def end_computation_timer(self):
        self.t_comp = (time.time() - self.iter_start_time) / self.batch_size
        # reduce computational time data point (avg) and send to the process of rank 0
        self.t_comp = comm.reduce(self.t_comp, average=True, all_reduce=False)

    def end_dataloading_timer(self):
        self.t_data = self.iter_start_time - self.iter_end_time # is it per sample or per batch?
        # reduce data loading per data point (avg) and send to the process of rank 0
        self.t_data = comm.reduce(self.t_data, average=True, all_reduce=False)

    def log_iter(self, learning_rates, losses, visuals): # TODO: implement visuals, tensorboard, wandb, python logger
        """Parameters:
            iters (int) -- current training iteration
            losses (tuple/list) -- training losses
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        # reduce losses (avg) and send to the process of rank 0
        losses = comm.reduce(losses, average=True, all_reduce=False)
        self.total_steps += self.batch_size * multi_gpu.get_world_size()

        if not multi_gpu.is_main_process():
            return 
        
        lr_G, lr_D = learning_rates

        message = '\n' + 20 * '-' + ' '
        message += '(iter: %d, samples: %d | comp: %.3f, data: %.3f | lr_G: %.7f, lr_D = %.7f)' \
                       % (self.iter_idx, self.total_steps, self.t_comp, self.t_data, lr_G, lr_D)

        message += ' ' + 20 * '-' +  '\n'
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)


