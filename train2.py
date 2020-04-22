import os
import time
import torch
from options.train_options import TrainOptions
from data import CustomDataLoader
from models import create_model
from util.distributed import multi_gpu, comm

def main():
    options = TrainOptions() # TODO: this is ugly as hell, used only for printing, make it nicer
    
    opt = options.parse()
    options.print_options(opt)
    is_main_process = True


    data_loader = CustomDataLoader(opt)
    model = create_model(opt)
    device = model.device

    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if is_main_process:
            lr_G, lr_D = model.get_learning_rate()
            print('\nlearning rates: lr_G = %.7f lr_D = %.7f' % (lr_G, lr_D))

        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            model.set_input(data)
            model.optimize_parameters()

            # at each step, every process goes through `batch_size` number of steps (data samples)
            steps_done = opt.batch_size # TODO: this is not the most accurate when data size is not a factor of batch size
            # at each iteration, provide each process with sum of number of steps done by each process
            steps_done = comm.reduce(steps_done, average=False, all_reduce=True, device=device)
            total_steps += steps_done
            epoch_iter += steps_done
            
                
            if epoch_iter % opt.print_freq == 0: # TODO: change so that it does it N number of times per epoch
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                # reduce losses (avg) and send to the process of rank 0
                losses = comm.reduce(losses, average=True, all_reduce=False, device=device)
                # reduce computational time and data loading per data point (avg) and send to the process of rank 0
                t_comp = comm.reduce(t_comp, average=True, all_reduce=False, device=device)
                t_data = comm.reduce(t_data, average=True, all_reduce=False, device=device)


            if is_main_process and total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        model.update_learning_rate()  # perform a scheduler step 

        if is_main_process:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            if epoch % opt.save_epoch_freq == 0:
                model.save_networks(epoch)

            epoch_time = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, epoch_time))
            

if __name__ == '__main__':
    main()
