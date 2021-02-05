from midaGAN.engines.base import BaseEngineWithInference
from midaGAN.utils import environment
from midaGAN.utils.builders import build_gan


class Inferer(BaseEngineWithInference):

    def __init__(self, conf):
        super().__init__(conf)
        environment.setup_logging_with_config(self.conf)
        self.model = build_gan(self.conf)

    def _set_mode(self):
        self.conf.mode = 'infer'
