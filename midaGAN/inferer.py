import os
import logging
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from monai.inferers import SlidingWindowInferer

from midaGAN.data import build_loader
from midaGAN.nn.gans import build_gan
from midaGAN.conf import init_config, InferenceConfig
from midaGAN.utils import io


class Inferer():
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)
        self.conf = self._build_config()
        self.data_loader = build_loader(self.conf)
        self.model = build_gan(self.conf)
        self.device = self.model.device

        self.sliding_window_inferer = SlidingWindowInferer(roi_size=(48, 224, 224),
                                                           sw_batch_size=1,
                                                           overlap=0.25)

    def run(self):
        has_metadata = False
        for data in self.data_loader:
            # Sometimes, metadata is necessary to be able to store the generated outputs.
            # E.g. origin, spacing and direction is required in order to properly save medical images.
            if isinstance(data, list): # dataloader yields a list when passing multiple values at once
                data, metadata = data
                metadata = [np.array(elem) for elem in metadata] # tensor -> numpy
                has_metadata = True

            data = data.to(self.device)
            # Sliding window (i.e. patch-wise) inference
            if self.sliding_window_inferer:
                out = self.sliding_window_inferer(data, self.predict)
            else:
                out = self.predict(data)
            
            if has_metadata:
                self._save(out, metadata)
            else:
                self._save(out)

    def predict(self, input):
        return self.model.infer(input)

    def _save(self, output, metadata):
        self.data_loader.dataset.save(output, metadata, self.output_dir)

    