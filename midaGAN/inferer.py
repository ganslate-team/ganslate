from monai.inferers import SlidingWindowInferer

class Inferer:
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.sliding_window_inferer = self._init_sliding_window_inferer()

    def infer(self, data, *args, **kwargs):
        data = data.to(self.model.device)
        # Sliding window (i.e. patch-wise) inference
        if self.sliding_window_inferer:
            return self.sliding_window_inferer(data, lambda x: self.model.infer(x, *args, **kwargs))
        return self.model.infer(data)

    def _init_sliding_window_inferer(self):
        if not self.conf.sliding_window:
            return
        return SlidingWindowInferer(roi_size=self.conf.sliding_window.window_size,
                                    sw_batch_size=self.conf.sliding_window.batch_size,
                                    overlap=self.conf.sliding_window.overlap,
                                    mode=self.conf.sliding_window.mode,
                                    cval=-1)
