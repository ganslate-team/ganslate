from midaGAN.engines.base import BaseEngine
from monai.inferers import SlidingWindowInferer

class Inferer(BaseEngine):
    def __init__(self, conf, model):
        super().__init__(conf)
        self.model = model
        self.sliding_window_inferer = self._init_sliding_window_inferer()

    def _set_mode(self):
        self.conf.mode = 'infer'

    def infer(self, data, *args, **kwargs):
        data = data.to(self.model.device)
        # Sliding window (i.e. patch-wise) inference
        if self.sliding_window_inferer:
            return self.sliding_window_inferer(data, lambda x: self.model.infer(x, *args, **kwargs))
        return self.model.infer(data)

    def _init_sliding_window_inferer(self):
        sw = self.conf[self.conf.mode].sliding_window
        if not sw:
            return
        return SlidingWindowInferer(roi_size=sw.window_size,
                                    sw_batch_size=sw.batch_size,
                                    overlap=sw.overlap,
                                    mode=sw.mode,
                                    cval=-1)
