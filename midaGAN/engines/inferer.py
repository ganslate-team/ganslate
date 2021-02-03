from midaGAN.engines.base import BaseEngineWithInference
from midaGAN.nn.gans import build_gan
from midaGAN.utils import environment


class Inferer(BaseEngineWithInference):
    def __init__(self, conf):
        super().__init__(conf)
        environment.setup_logging_with_config(self.conf)
        self.model = build_gan(self.conf)

    def _set_mode(self):
        self.conf.mode = 'infer'

    def _override_conf(self):
        ...
