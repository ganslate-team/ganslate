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
            # Sometimes, metadata is necessary to be able to store the generated outputs.
            # E.g. origin, spacing and direction is required in order to properly save medical images.

            input = data["input"]
            metadata = None
            if "metadata" in data:
                metadata = decollate(data["metadata"])

            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()
            out = self.infer(input)
            self.tracker.end_computation_timer()

            self.tracker.start_saving_timer()
            # Inference-time dataset class has to have a `save()` method
            if metadata:
                self.data_loader.dataset.save(out, metadata, self.output_dir / "output")
            else:
                self.data_loader.dataset.save(out, self.output_dir / "output")
            self.tracker.end_saving_timer()

            self.tracker.log_message(i, len_dataset=len(self.data_loader.dataset))
            self.tracker.start_dataloading_timer()
