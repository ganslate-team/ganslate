from midaGAN.engines.base import BaseEngineWithInference
from midaGAN.utils import environment
from midaGAN.utils.builders import build_gan, build_loader
from midaGAN.utils.io import decollate
from midaGAN.utils.trackers.inference import InferenceTracker


class Inferer(BaseEngineWithInference):

    def __init__(self, conf):
        super().__init__(conf)

        # Logging, dataloader and tracker only when not in deployment mode
        if not self.conf.infer.is_deployment:
            assert self.conf.infer.dataset, "Please specify the dataset for inference."
            environment.setup_logging_with_config(self.conf)
            self.tracker = InferenceTracker(self.conf)
            self.data_loader = build_loader(self.conf)

        self.model = build_gan(self.conf)

    def _set_mode(self):
        self.conf.mode = 'infer'

    def run(self):
        assert not self.conf.infer.is_deployment, \
            "`Inferer.run()` cannot be used in deployment, please use `Inferer.infer()`."

        self.logger.info("Inference started.")

        self.tracker.start_dataloading_timer()
        for i, data in enumerate(self.data_loader, start=1):
            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()
            out = self.infer(data["input"])
            self.tracker.end_computation_timer()

            self.tracker.start_saving_timer()
            # Inference-time dataset class has to have a `save()` method
            save_dir = self.output_dir / "saved"
            if "metadata" in data:
                self.data_loader.dataset.save(out, save_dir, metadata=decollate(data["metadata"]))
            else:
                self.data_loader.dataset.save(out, save_dir)
            self.tracker.end_saving_timer()

            self.tracker.log_message(i, len_dataset=len(self.data_loader.dataset))
            self.tracker.start_dataloading_timer()
