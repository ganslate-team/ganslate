import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import wandb

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save_networks('latest')
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)

        epoch_time = time.time() - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, epoch_time))
        
        if opt.wandb:
            epoch_log_dict = {'epoch_time': epoch_time,
                              'learning_rate': model.get_learning_rate()}
            # get all the losses
            for k, v in losses.items():
                epoch_log_dict['loss_%s' % k] = v

            # get the inputs and outputs of the network
            visuals = model.get_current_visuals()
            for k, v in visuals.items():
                # keys are e.g. real_A, fake_B and values are their corresponding volumes
                v = v[0].permute(1,2,3,0) # take one from the batch and CxHxWxL -> HxWxLxC
                v = v.cpu().detach().numpy()
                epoch_log_dict[k] = [wandb.Image(_slice) for _slice in v]

            # log all the information of the epoch
            wandb.log(epoch_log_dict)

        model.update_learning_rate()
