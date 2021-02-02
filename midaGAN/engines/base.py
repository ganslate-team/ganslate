from abc import ABC, abstractmethod
import copy


class BaseEngine(ABC):
    def __init__(self, conf):
        # deep copy to isolate the conf.mode of an engine from other engines (e.g train from val)
        self.conf = copy.deepcopy(conf)
        self._set_mode()

    @abstractmethod
    def _set_mode(self):
        """Sets the mode for the particular engine.
        E.g., 'train' for Trainer, 'val' for 'Validator' etc."""
        self.conf.mode = ...
