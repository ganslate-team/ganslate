import os
import time
import torch
from options.train_options import TrainOptions
from data import build_loader
from models import build_model
from util.visualizer import Visualizer
from util.distributed import multi_gpu, comm, init_distributed
import logging

logger = logging.getLogger(__name__)

def print_info(opt, options, model, data_loader):
    options.print_options(opt)
    print("dataset [%s] was created" % (type(data_loader.dataset).__name__))
    print('len(A),len(B)=', data_loader.dataset.A_size, data_loader.dataset.B_size)
    print('# of training pairs = %d' % len(data_loader))
    print("model [%s] was created" % (type(model).__name__))
    model.print_networks()
    print('Invertible layers memory saving: {}'.format('ON' if opt.use_memory_saving else 'OFF'))
    print('Distributed data parallel training: {}'.format('ON' if opt.distributed else 'OFF'))
    num_devices = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
    print('Batch size per GPU: {}'.format(opt.batch_size // num_devices))

def main():
    options = TrainOptions() # TODO: this is ugly as hell, used only for printing, make it nicer
    opt = options.parse()
    is_main_process = True

    if opt.distributed:
        init_distributed(opt.local_rank)
        is_main_process = multi_gpu.get_rank() == 0

    data_loader = build_loader(opt)
    model = build_model(opt)
    device = model.device

    if is_main_process:
        print_info(opt, options, model, data_loader)
        if opt.wandb:
            import wandb
            wandb.init(project="gan-translation", entity="maastro-clinic")
        visualizer = Visualizer(opt)

    total_steps = 0
    for epoch in range(opt.continue_iter, opt.n_iters + opt.n_iters_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if is_main_process:
            lr_G, lr_D = model.get_learning_rates()
            print('\nlearning rates: lr_G = %.7f lr_D = %.7f' % (lr_G, lr_D))

        #if opt.distributed:
        #    data_loader.sampler.set_epoch(epoch) # so that DistributedSampler shuffles properly

        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            model.set_input(data)
            model.optimize_parameters()

            # at each step, every process goes through `batch_size` number of steps (data samples)
            steps_done = opt.batch_size # TODO: this is not the most accurate when data size is not a factor of batch size
            # at each iteration, provide each process with sum of number of steps done by each process
            steps_done = comm.reduce(steps_done, average=False, all_reduce=True)
            total_steps += steps_done
            epoch_iter += steps_done
            
            if is_main_process:
                visualizer.reset()
                
            if epoch_iter % opt.print_freq == 0: # TODO: change so that it does it N number of times per epoch
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                # reduce losses (avg) and send to the process of rank 0
                losses = comm.reduce(losses, average=True, all_reduce=False)
                # reduce computational time and data loading per data point (avg) and send to the process of rank 0
                t_comp = comm.reduce(t_comp, average=True, all_reduce=False)
                t_data = comm.reduce(t_data, average=True, all_reduce=False)

                if is_main_process:
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if is_main_process and total_steps % opt.update_html_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if is_main_process and total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save_checkpoint('latest')

            iter_data_time = time.time()

        model.update_learning_rate()  # perform a scheduler step 

        if is_main_process:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_checkpoint('latest')
            if epoch % opt.save_epoch_freq == 0:
                model.save_checkpoint(epoch)

            epoch_time = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.n_iters + opt.n_iters_decay, epoch_time))
            
            if opt.wandb:
                # TODO: do this properly and in a separate function
                epoch_log_dict = {'epoch_time': int(epoch_time),
                                  'lr_G': lr_G,
                                  'lr_D': lr_D}
                # get all the losses
                for k, v in losses.items():
                    epoch_log_dict['loss_%s' % k] = v

                # get the inputs and outputs of the network
                visuals = model.get_current_visuals()
                for k, v in visuals.items():
                    # keys are e.g. real_A, fake_B and values are their corresponding volumes
                    v = v[0].permute(1,2,3,0) # take one from the batch and CxLxHxW -> LxHxWxC
                    v = v.cpu().detach().numpy()
                    epoch_log_dict[k] = [wandb.Image(_slice) for _slice in v]

                # log all the information of the epoch
                wandb.log(epoch_log_dict)

if __name__ == '__main__':
    main()
