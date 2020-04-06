import os
import time
import torch
from options.train_options import TrainOptions
from data import CustomDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.distributed import multi_gpu
import logging

logger = logging.getLogger(__name__)

def main():
    options = TrainOptions()
    opt = options.parse()
    is_main_process = True

    # Setup distributed computing
    # This is set by the parallel computation script (launch.py)
    if opt.distributed:
        num_gpu = int(os.environ.get('WORLD_SIZE', 1))
        if num_gpu > 1:
            torch.cuda.set_device(opt.local_rank)
            torch.distributed.init_process_group(
                backend='nccl', init_method='env://'
            )
            multi_gpu.synchronize()
            logger.info(f'Number of GPUs available in world: {num_gpu}.')
            is_main_process = multi_gpu.get_rank() == 0
            opt.gpu_ids = [opt.local_rank]
        else:
            opt.distributed = False

    data_loader = CustomDataLoader(opt)

    model = create_model(opt)
    model.setup(opt)
    
    if is_main_process:
        # ------------ Print info ------------
        options.print_options(opt)
        print("dataset [%s] was created" % (data_loader.dataset.name()))
        print('len(A),len(B)=', data_loader.dataset.A_size, data_loader.dataset.B_size)
        print('# of training pairs = %d' % len(data_loader))
        print("model [%s] was created" % (model.name()))
        model.print_networks(opt.verbose)
        print('Invertible layers memory saving: {}'.format('ON' if not opt.use_naive else 'OFF'))
        print('Distributed data parallel training: {}'.format('ON' if opt.distributed else 'OFF'))
        print('Batch size per GPU: {}'.format(opt.batchSize // len(opt.gpu_ids)))
        # ------------------------------------

        if opt.wandb:
            import wandb
            wandb.init(project="gan-translation", entity="maastro-clinic")
    
        visualizer = Visualizer(opt)

    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if is_main_process:
            lr_G, lr_D = model.get_learning_rate()
            print('\nlearning rates: lr_G = %.7f lr_D = %.7f' % (lr_G, lr_D))
        if opt.distributed:
            data_loader.sampler.set_epoch(epoch) # so that DistributedSampler shuffles properly

        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if is_main_process:
                visualizer.reset()
                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                    model.save_networks('latest')

            iter_data_time = time.time()

        if is_main_process:    
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            if epoch % opt.save_epoch_freq == 0:
                model.save_networks(epoch)

            epoch_time = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, epoch_time))
            
            if opt.wandb:
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

        model.update_learning_rate()

if __name__ == '__main__':
    main()
