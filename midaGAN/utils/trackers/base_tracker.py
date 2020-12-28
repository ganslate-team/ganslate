import time
from pathlib import Path
from omegaconf import OmegaConf
from midaGAN.utils import communication


class BaseTracker:
    """"Base for training and inference trackers."""

    def __init__(self, conf):
        self.batch_size = conf.batch_size
        self.output_dir = conf.logging.checkpoint_dir if conf.is_train else conf.logging.inference_dir
        self.iter_idx = None
        self.iter_end_time = None
        self.iter_start_time = None
        self.t_data = None
        self.t_comp = None

        self.save_config(conf)

    def save_config(self, conf):
        if communication.get_local_rank() == 0:
            mode = "training" if conf.is_train else "inference"
            config_path = Path(self.output_dir) / f"{mode}_config.yaml"
            with open(config_path, "w") as file:
                file.write(OmegaConf.to_yaml(conf))

    def set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx

    def start_computation_timer(self):
        self.iter_start_time = time.time()

    def start_dataloading_timer(self):
        self.iter_end_time = time.time()

    def end_computation_timer(self):
        self.t_comp = (time.time() - self.iter_start_time) / self.batch_size
        # reduce computational time data point (avg) and send to the process of rank 0
        self.t_comp = communication.reduce(self.t_comp, average=True, all_reduce=False)

    def end_dataloading_timer(self):
        self.t_data = self.iter_start_time - self.iter_end_time  # is it per sample or per batch?
        # reduce data loading per data point (avg) and send to the process of rank 0
        self.t_data = communication.reduce(self.t_data, average=True, all_reduce=False)
