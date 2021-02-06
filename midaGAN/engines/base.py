import copy
from abc import ABC, abstractmethod
from pathlib import Path
import logging

from midaGAN.utils import sliding_window_inferer


class BaseEngine(ABC):

    def __init__(self, conf):
        # deep copy to isolate the conf.mode of an engine from other engines (e.g train from val)
        self.conf = copy.deepcopy(conf)
        self._set_mode()
        self._override_conf()

        self.model = None
        self.logger = logging.getLogger(type(self).__name__)

    @abstractmethod
    def _set_mode(self):
        """Sets the mode for the particular engine.
        E.g., 'train' for Trainer, 'val' for 'Validator' etc."""
        self.conf.mode = ...

    def _override_conf(self):
        """If any default overriding of config necessary, define it
        in this method when inherited."""


class BaseEngineWithInference(BaseEngine):

    def __init__(self, conf):
        super().__init__(conf)
        self.sliding_window_inferer = self._init_sliding_window_inferer()

    def infer(self, data, *args, **kwargs):
        data = data.to(self.model.device)
        # Sliding window (i.e. patch-wise) inference
        if self.sliding_window_inferer:
            return self.sliding_window_inferer(data, self.model.infer, *args, **kwargs)

        return self.model.infer(data)

    def _init_sliding_window_inferer(self):
        sw = self.conf[self.conf.mode].sliding_window

        if not sw:
            return None

        return sliding_window_inferer.SlidingWindowInferer(roi_size=sw.window_size,
                                                           sw_batch_size=sw.batch_size,
                                                           overlap=sw.overlap,
                                                           mode=sw.mode,
                                                           cval=-1)
