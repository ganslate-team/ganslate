import os
import time
import torch
from data import CustomDataLoader
from models import create_model
from util.distributed import multi_gpu, comm

from omegaconf import OmegaConf
from conf.config import Config

def main():
    conf = OmegaConf.structured(Config)
    cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli)

    print(conf.pretty())
    
    is_main_process = True

    data_loader = CustomDataLoader(conf)
    model = create_model(conf)
    device = model.device

    total_steps = 0
    for epoch in range(conf.continue_epoch, conf.n_epochs + conf.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if is_main_process:
            lr_G, lr_D = model.get_learning_rate()
            print('\nlearning rates: lr_G = %.7f lr_D = %.7f' % (lr_G, lr_D))

        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_steps % conf.logging.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            model.set_input(data)
            model.optimize_parameters()

            # at each step, every process goes through `batch_size` number of steps (data samples)
            steps_done = conf.batch_size # TODO: this is not the most accurate when data size is not a factor of batch size
            # at each iteration, provide each process with sum of number of steps done by each process
            steps_done = comm.reduce(steps_done, average=False, all_reduce=True, device=device)
            total_steps += steps_done
            epoch_iter += steps_done
            
                
            if epoch_iter % conf.logging.print_freq == 0: # TODO: change so that it does it N number of times per epoch
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / conf.batch_size

                # reduce losses (avg) and send to the process of rank 0
                losses = comm.reduce(losses, average=True, all_reduce=False, device=device)
                # reduce computational time and data loading per data point (avg) and send to the process of rank 0
                t_comp = comm.reduce(t_comp, average=True, all_reduce=False, device=device)
                t_data = comm.reduce(t_data, average=True, all_reduce=False, device=device)

            iter_data_time = time.time()

        model.update_learning_rate()  # perform a scheduler step 

        if is_main_process:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            if epoch % conf.logging.save_epoch_freq == 0:
                model.save_networks(epoch)

            epoch_time = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, conf.n_epochs + conf.n_epochs_decay, epoch_time))
            

if __name__ == '__main__':
    main()
