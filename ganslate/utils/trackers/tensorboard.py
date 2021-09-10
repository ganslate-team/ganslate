from torch.utils.tensorboard import SummaryWriter
from ganslate.utils.trackers.utils import process_visuals_wandb_tensorboard


class TensorboardTracker:

    def __init__(self, conf):
        self.writer = SummaryWriter(conf[conf.mode].output_dir)
        self.image_window = conf[conf.mode].logging.image_window

    def close(self):
        self.writer.close()

    def log_iter(self,
                 iter_idx,
                 visuals,
                 mode,
                 learning_rates=None,
                 losses=None,
                 metrics=None):
        # Learning rates
        if learning_rates is not None:
            for name, learning_rate in learning_rates.items():
                self.writer.add_scalar(f"Learning Rates/{name}", learning_rate, iter_idx)

        # Losses
        if losses is not None:
            for name, loss in losses.items():
                self.writer.add_scalar(f"Losses/{name}", loss, iter_idx)

        # Metrics
        if metrics is not None:
            for name, metric in metrics.items():
                self.writer.add_scalar(f"Metrics ({mode})/{name}", metric, iter_idx)

        # Normal images
        normal_visuals = process_visuals_wandb_tensorboard(visuals, image_window=None)
        self._log_images(iter_idx, normal_visuals, tag=mode)

        # Windowed images
        if self.image_window:
            windowed_visuals = process_visuals_wandb_tensorboard(visuals, self.image_window)
            self._log_images(iter_idx, windowed_visuals, tag=f"{mode}_windowed")

    def _log_images(self, iter_idx, visuals, tag):
        visuals = visuals if isinstance(visuals, list) else [visuals]
        for idx, visual in enumerate(visuals):
            name, image = visual['name'], visual['image']
            name = f"{idx}_{name}" if len(visuals) > 1 else name
            self.writer.add_image(f"{tag}/{name}", image, iter_idx, dataformats='HWC')
