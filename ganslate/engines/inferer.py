from loguru import logger

from ganslate.engines.base import BaseEngineWithInference
from ganslate.utils import environment
from ganslate.utils.builders import build_gan, build_loader
from ganslate.utils.trackers.inference import InferenceTracker
from ganslate.utils import communication


class Inferer(BaseEngineWithInference):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logger

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
        for i, data in enumerate(self.data_loader):
            # Iteration index
            # (1) When using DDP, multiply with world size since each process does an iteration
            # (2) Multiply with batch size to get accurate info on how many examples are done
            # (3) Add 1 to start from iter 1 instead of 0
            iter_idx = i * communication.get_world_size() * self.conf.infer.batch_size + 1
            self.tracker.set_iter_idx(iter_idx)
            if i == 0:
                input_key = self._get_input_key(data)
                if not hasattr(self.data_loader.dataset, "save"):
                    self.logger.warning(
                        "The dataset class used does not have a 'save' method."
                        " It is not necessary, however, it may be useful in cases"
                        " where the outputs should be stored individually"
                        " ('images/' folder saves input and output in a single image), "
                        " or in a specific format.")

            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()
            out = self.infer(data[input_key])
            self.tracker.end_computation_timer()

            self.tracker.start_saving_timer()
            # Save the output as specified in dataset`s `save` method, if implemented
            metadata = data["metadata"] if "metadata" in data else None
            self.save_generated_tensor(generated_tensor=out,
                                       metadata=metadata,
                                       data_loader=self.data_loader)
            self.tracker.end_saving_timer()

            visuals = {"input": data[input_key], "output": out.cpu()}
            self.tracker.log_iter(visuals, len(self.data_loader.dataset))
            self.tracker.start_dataloading_timer()
        self.tracker.close()

    def _get_input_key(self, data):
        """The dataset (dataloader) needs to return a dict with input data 
        either under the key 'input' or 'A'."""
        if "input" in data:
            return "input"
        elif "A" in data:
            return "A"
        else:
            raise ValueError("An inference dataset needs to provide"
                             "the input data under the dict key 'input' or 'A'.")
