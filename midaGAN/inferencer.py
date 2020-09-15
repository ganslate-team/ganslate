import os
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

import midaGAN
from midaGAN.data import build_loader
from midaGAN.nn.gans import build_gan
from midaGAN.conf import init_config


class Inferencer():
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)
        conf = self._build_config()
        print(OmegaConf.to_yaml(conf))

        self.data_loader = build_loader(conf)
        self.model = build_gan(conf)

    def infer(self, input):
        for data in self.data_loader:
            out = self.model.infer(input)
            self._save(out)

    def _save(self, output):
        self.data_loader.dataset.save(out)

    def _build_config(self):
        # Load the inference configuration
        cli = OmegaConf.from_cli()
        if cli.config:
            inference_conf = OmegaConf.load(cli.config)
            cli.pop("config")
            inference_conf = OmegaConf.merge(inference_conf, cli)
        else:
            inference_conf = cli
        # Perform init_config to validate the config against the config dataclasses
        inference_conf = init_config(inference_conf, config_class=midaGAN.conf.config.InferenceConfig)

        # Fetch the config that was used during training of this specific run
        train_conf = Path(inference_conf.checkpoint_dir) / "config.yaml"
        train_conf = OmegaConf.load(str(train_conf))

        # Override training config with inference-specific params
        train_conf.load_iter = inference_conf.load_iter
        train_conf.dataset = dict(inference_conf.dataset)
        train_conf.dataset.shuffle = False
        train_conf.gan.is_train = False
        
        return init_config(train_conf)
