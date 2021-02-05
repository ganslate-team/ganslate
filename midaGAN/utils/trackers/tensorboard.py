from torch.utils.tensorboard import SummaryWriter


class TensorboardTracker:

    def __init__(self, conf):
        self.writer = SummaryWriter(conf[conf.mode].output_dir)

    def close(self):
        self.writer.close()

    def log_iter(self, iter_idx, learning_rates, losses, visuals, metrics):
        # TODO: metrics unused
        # Learning rates
        lr_G, lr_D = learning_rates["lr_G"], learning_rates["lr_D"]
        self.writer.add_scalar('Learning Rates/lr_G', lr_G, iter_idx)
        self.writer.add_scalar('Learning Rates/lr_D', lr_D, iter_idx)

        # Losses
        for name, loss in losses.items():
            self.writer.add_scalar(f"Losses/{name}", loss, iter_idx)

        # Image
        name, image = visuals['name'], visuals['image']
        self.writer.add_image(f"Visuals/{name}", image, iter_idx, dataformats='CHW')
