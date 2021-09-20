import copy
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger

from ganslate.utils import sliding_window_inferer
from ganslate.utils.io import decollate


class BaseEngine(ABC):

    def __init__(self, conf):
        # deep copy to isolate the conf.mode of an engine from other engines (e.g train from val)
        self.conf = copy.deepcopy(conf)
        self._set_mode()

        self.output_dir = Path(conf[conf.mode].output_dir) / self.conf.mode
        self.model = None
        self.logger = logger

    @abstractmethod
    def _set_mode(self):
        """Sets the mode for the particular engine.
        E.g., 'train' for Trainer, 'val' for 'Validator' etc."""
        self.conf.mode = ...


class BaseEngineWithInference(BaseEngine):

    def __init__(self, conf):
        super().__init__(conf)
        self.sliding_window_inferer = self._init_sliding_window_inferer()

    def infer(self, data, *args, **kwargs):
        data = data.to(self.model.device)
        # Sliding window (i.e. patch-wise) inference
        if self.sliding_window_inferer:
            return self.sliding_window_inferer(data, self.model.infer, *args, **kwargs)
        return self.model.infer(data, *args, **kwargs)

    def _init_sliding_window_inferer(self):
        sw = self.conf[self.conf.mode].sliding_window
        if not sw:
            return None

        return sliding_window_inferer.SlidingWindowInferer(roi_size=sw.window_size,
                                                           sw_batch_size=sw.batch_size,
                                                           overlap=sw.overlap,
                                                           mode=sw.mode,
                                                           cval=-1)

    def save_generated_tensor(self,
                              generated_tensor,
                              metadata,
                              data_loader,
                              idx=None,
                              dataset_name=None):
        # A dataset object has to have a `save()` method if it
        # wishes to save the outputs in a particular way or format
        save_fn = getattr(data_loader.dataset, "save", False)
        if save_fn:
            # Tolerates `save` methods with and without `metadata` argument
            def save(tensor, save_dir, metadata=None):
                if metadata is None:
                    save_fn(tensor=tensor, save_dir=save_dir)
                else:
                    save_fn(tensor=tensor, save_dir=save_dir, metadata=metadata)

            # Output dir
            save_dir = "saved/"
            if dataset_name is not None:
                save_dir += f"{dataset_name}/"
            if idx is not None:
                save_dir += f"{idx}/"
            save_dir = self.output_dir / save_dir

            # Metadata
            if metadata:
                # After decollate, it is a list of length equal to batch_size,
                # containing separate metadata for each tensor in the mini-batch
                metadata = decollate(metadata, batch_size=len(generated_tensor))

            # Loop over the batch and save each tensor
            for batch_idx in range(len(generated_tensor)):
                tensor = generated_tensor[batch_idx]
                current_metadata = metadata[batch_idx] if metadata is not None else metadata
                save(tensor=tensor, save_dir=save_dir, metadata=current_metadata)
