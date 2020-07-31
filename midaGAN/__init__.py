import logging


class midaGanBase:
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(type(self).__name__)